"""Live plate tracker when the plate is nudged: pipette1 clips with the plate slid / bumped / turned a little.

Each sample: --seconds of consecutive pipette1 session-1 frames (plate clear at the start, as the demo) fed to a fresh
PlateTracker. Three nudges per clip, each a rigid move of the whole frame about the plate centre (random direction,
seeded): a slow slide (0.7 well over 0.5 s), a bump (1.2 wells in 2 frames), a turn (4 deg over 0.2 s). The plate
stays where it was pushed. Reference: the tracker's own grid on the unmoved video
(data/processed/dispense_eval/<ref>.npz, checked: no wrong stretches), moved the same way.
Per nudge: frames after it starts until the grid is back within 0.25 spacing (recover), frames with a grid more than
0.5 spacing off (wrong), frames without a grid (none).

    python scripts/eval_plate_nudge.py --samples 12
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import grid_error, pitch_px, project  # noqa: E402
from pipette_coach.localizer.live import PlateTracker  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

FPS = 30
# (name, start s, frames to ramp over, move in wells, turn deg)
NUDGES = [("slide", 1.5, 15, 0.7, 0.0), ("bump", 4.0, 2, 1.2, 0.0), ("turn", 6.5, 6, 0.0, 4.0)]


def rigid(centre: np.ndarray, d: np.ndarray, deg: float) -> np.ndarray:
    A = np.vstack([cv2.getRotationMatrix2D(tuple(map(float, centre)), deg, 1.0), [0, 0, 1]])
    A[:2, 2] += d
    return A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=8.5)
    ap.add_argument("--ref", default="anchor3")
    ap.add_argument("--kw", default="{}", help="PlateTracker settings as JSON")
    ap.add_argument("--name", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", type=float, help="only the clip starting at this time (s): print every frame")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    ref = np.load(PROCESSED_DIR / "dispense_eval" / f"{a.ref}.npz")
    Href, tref = ref["H"], ref["t"]
    tips = np.load(PROCESSED_DIR / "dispense_eval" / f"{a.ref}__all.npz")["tips"][:, 0, 2]
    n = int(a.seconds * FPS)
    ok_all, clear = np.isfinite(Href[:, 0, 0]), ~(np.nan_to_num(tips) >= 0.1)
    starts = [k for k in range(0, len(tref) - n, 15)
              if ok_all[k : k + n].all() and clear[k : k + 15].all() and tref[k + n] < 384]  # fmt: skip
    cap = cv2.VideoCapture(str(REPO_ROOT / "data" / "raw" / "pipette1.mp4"))
    rows = []
    for s_i in rng.choice(starts, a.samples, replace=False):
        if a.trace is not None and abs(tref[s_i] - a.trace) > 0.01:
            rng.uniform(0, 2 * np.pi, len(NUDGES)), rng.choice([-1, 1], len(NUDGES))  # same draws as the full run
            continue
        dirs = rng.uniform(0, 2 * np.pi, len(NUDGES))
        turns = rng.choice([-1, 1], len(NUDGES))
        tr = PlateTracker(cfg, det, net, verify_every=30, **json.loads(a.kw))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tref[s_i] * FPS)))
        A, errs = np.eye(3), []
        for k in range(n):
            ok, f = cap.read()
            if not ok:
                break
            H0 = Href[s_i + k]
            p = pitch_px(H0)
            for j, (_, t0, ramp, mv, deg) in enumerate(NUDGES):
                k0 = int(t0 * FPS)
                if k0 <= k < k0 + ramp:  # one more step of this nudge (about the plate centre as now seen)
                    cc = project(A @ H0, np.array([[5.5, 3.5]]))[0]
                    step = mv * p * np.array([np.cos(dirs[j]), np.sin(dirs[j])]) / ramp
                    A = rigid(cc, step, turns[j] * deg / ramp) @ A
            g = cv2.warpPerspective(f, A, f.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(40, 40, 40))
            out = tr.update(g)
            errs.append(np.nan if out["H"] is None else grid_error(A @ H0, out["H"], ignore_labelling=False))
            if a.trace is not None:
                geo = np.nan if out["H"] is None else grid_error(A @ H0, out["H"])
                print(f"{k:4d} err {errs[-1]:5.2f} (wells only {geo:5.2f})  {out['status']}")
        e = np.array(errs)
        first = int(np.argmax(np.isfinite(e))) if np.isfinite(e).any() else None
        row = {"start_s": float(tref[s_i]), "first_lock": first,
               "wrong_before": int((e[: int(NUDGES[0][1] * FPS)] > 0.5).sum())}  # fmt: skip
        for name, t0, ramp, _, _ in NUDGES:
            k0 = int(t0 * FPS)
            w = e[k0 : k0 + int(2.5 * FPS)]
            good = np.where(w < 0.25)[0]
            back = [g for g in good if g >= ramp and (w[g : g + 5] < 0.25).all()]
            row[name] = {"recover": int(back[0]) if back else None, "wrong": int((w > 0.5).sum()),
                         "none": int(np.isnan(w).sum())}  # fmt: skip
        rows.append(row)
        print(row, flush=True)
    m = {"samples": len(rows), "kw": json.loads(a.kw), "nudges": NUDGES}
    for name, *_ in NUDGES:
        r = [row[name] for row in rows]
        rec = [x["recover"] for x in r if x["recover"] is not None]
        m[name] = {"recovered": len(rec) / len(r), "recover_frames_p50": float(np.median(rec)) if rec else None,
                   "recover_frames_max": max(rec) if rec else None,
                   "wrong_frames": int(sum(x["wrong"] for x in r)),
                   "no_grid_frames": int(sum(x["none"] for x in r))}  # fmt: skip
    m["rows"] = rows
    run = new_run_dir("plate-nudge" + (f"-{a.name}" if a.name else ""))
    write_metrics(run, m)
    (run / "notes.md").write_text(
        "# Live plate tracker with the plate nudged (pipette1)\n\n"
        f"{len(rows)} clips x {a.seconds} s; nudges {NUDGES} (name, start s, ramp frames, wells, deg). "
        "recover = frames from the nudge start until the grid is within 0.25 spacing for 5 frames (window 2.5 s); "
        "wrong = frames > 0.5 spacing off; no_grid = frames without a grid.\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in m.items() if k != "rows")
        + "\n\nKeep local (third-party video).\n"
    )
    print({k: v for k, v in m.items() if k != "rows"}, "->", run)


if __name__ == "__main__":
    main()
