"""Live plate tracker under other cameras: pipette1.mp4 clips re-filmed by random virtual cameras.

A new webcam sees the plate at another size, rotation and tilt than pipette1. Each sample: 1.5 s of consecutive
pipette1 frames (starting with the plate clear, as the demo does; hands and pipette may come in later) warped by
one random camera - scale 0.5-1.6x (log-uniform; one well spacing ~15-46 px), any in-plane rotation, tilt
0-max_tilt deg, random position in a 1920 x 1080 frame - and fed to a fresh PlateTracker frame by frame.
Reference: the tracker's own grid on the unwarped video
(data/processed/dispense_eval/<ref>.npz, checked: no wrong stretches), warped the same way and labelled by the same
A1 rule (configs/localizer.yaml), so this tests the tracker, not the rule. Per sample: locked? frames from the first
lock with the grid within 0.25 spacing (correct) / more than 0.5 off (wrong).

    python scripts/eval_plate_cameras.py --samples 40
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_generalization import camera_warp  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import GRID, grid_error, orient, project  # noqa: E402
from pipette_coach.localizer.live import PlateTracker  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

OUT_W, OUT_H = 1920, 1080


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--max-tilt", type=float, default=20.0)
    ap.add_argument("--ref", default="anchor3")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--no-lock-checks", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kw", default="{}", help="more PlateTracker settings as JSON")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    oc = cfg["orientation"]
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    ref = np.load(PROCESSED_DIR / "dispense_eval" / f"{a.ref}.npz")
    Href, tref = ref["H"], ref["t"]
    cap = cv2.VideoCapture(str(REPO_ROOT / "data" / "raw" / "pipette1.mp4"))
    ok_all = np.isfinite(Href[:, 0, 0])
    # start like the live demo: plate in view, no pipette over it yet (no tip >= 0.1 in the first 15 frames), and
    # inside session 1 (the plate is moved after ~386 s)
    tips = np.load(PROCESSED_DIR / "dispense_eval" / f"{a.ref}__all.npz")["tips"][:, 0, 2]
    clear = ~(np.nan_to_num(tips) >= 0.1)
    starts = [k for k in range(0, len(tref) - a.frames, 15)
              if ok_all[k : k + a.frames].all() and clear[k : k + 15].all() and tref[k + a.frames] < 384]  # fmt: skip
    rows, t0, bad_imgs = [], time.time(), []
    for s_i in rng.choice(starts, a.samples, replace=False):
        s = float(np.exp(rng.uniform(np.log(0.5), np.log(1.6))))
        rot, tilt, tdir = rng.uniform(0, 360), rng.uniform(0, a.max_tilt), rng.uniform(0, 360)
        H0 = Href[s_i]
        centre = project(H0, np.array([[5.5, 3.5]]))[0]
        corners = project(H0, np.array([[-2, -2], [13, -2], [13, 9], [-2, 9]], float)) - centre
        half = np.full(2, np.linalg.norm(corners, axis=1).max() * s * 1.05)  # any rotation stays in the frame
        dst = np.array([rng.uniform(min(half[0], OUT_W / 2), max(OUT_W - half[0], OUT_W / 2)),
                        rng.uniform(min(half[1], OUT_H / 2), max(OUT_H - half[1], OUT_H / 2))])  # fmt: skip
        Wp = camera_warp(centre, s, rot, tilt, tdir, dst)
        moved = dst - project(Wp, centre[None])[0]  # the tilt also shifts the centre: put it back at dst
        Wp = np.array([[1, 0, moved[0]], [0, 1, moved[1]], [0, 0, 1]]) @ Wp
        tr = PlateTracker(
            cfg, det, net, verify_every=30, anchor=not a.no_anchor, lock_checks=not a.no_lock_checks, **json.loads(a.kw)
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tref[s_i] * 30)))
        errs, geo = [], []
        for k in range(a.frames):
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.warpPerspective(f, Wp, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderValue=(90, 90, 90))
            out = tr.update(g)
            H_true = orient(Wp @ Href[s_i + k], oc["a1_rule"], oc["camera_mirrored"])
            errs.append(None if out["H"] is None else grid_error(H_true, out["H"], ignore_labelling=False))
            geo.append(None if out["H"] is None else grid_error(H_true, out["H"]))  # wells right, A1 ignored
        got = [e for e in errs if e is not None]
        first = next((i for i, e in enumerate(errs) if e is not None), None)
        row = {"scale": s, "rot": rot, "tilt": tilt, "locked": first is not None, "first_lock": first,
               "correct": sum(e < 0.25 for e in got) / max(len(errs) - (first or 0), 1) if got else 0.0,
               "wrong": sum(e > 0.5 for e in got) / max(len(got), 1),
               "wrong_wells": sum(e > 0.5 for e in geo if e is not None) / max(len(got), 1)}  # fmt: skip
        rows.append(row)
        if row["wrong"] > 0:  # last frame: model grid green, reference red (A1 = big dot)
            for H, col in ((H_true, (0, 0, 255)), (out["H"], (0, 255, 0))):
                if H is not None:
                    for q, (u, v) in enumerate(project(H, GRID)):
                        cv2.circle(g, (int(u), int(v)), 9 if q == 0 else 3, col, -1)
            cv2.putText(g, f"s={s:.2f} rot={rot:.0f} tilt={tilt:.0f}", (20, 50), 0, 1.5, (0, 255, 255), 3)
            bad_imgs.append((cv2.resize(g, (960, 540)), row))
        print(f"s={s:.2f} rot={rot:5.1f} tilt={tilt:4.1f}: locked at {first}, correct {row['correct']:.0%}, "
              f"wrong {row['wrong']:.0%} (wells {row['wrong_wells']:.0%}) ({time.time() - t0:.0f}s)",
              flush=True)  # fmt: skip
    sc = np.array([r["scale"] for r in rows])
    tl = np.array([r["tilt"] for r in rows])

    def agg(m):
        rr = [r for r, k in zip(rows, m, strict=True) if k]
        return {"n": len(rr), "locked": float(np.mean([r["locked"] for r in rr])),
                "correct_after_lock": float(np.mean([r["correct"] for r in rr])),
                "wrong_share": float(np.mean([r["wrong"] for r in rr])),
                "wrong_wells_share": float(np.mean([r["wrong_wells"] for r in rr])),
                "any_wrong": float(np.mean([r["wrong"] > 0 for r in rr]))} if rr else None  # fmt: skip

    m = {"samples": len(rows), "frames": a.frames, "max_tilt": a.max_tilt, "anchor": not a.no_anchor, "kw": a.kw,
         "lock_checks": not a.no_lock_checks,
         "all": agg(np.ones(len(rows), bool)),
         "scale_0.5-0.8": agg((sc >= 0.5) & (sc < 0.8)), "scale_0.8-1.25": agg((sc >= 0.8) & (sc < 1.25)),
         "scale_1.25-1.6": agg(sc >= 1.25), "tilt_0-10": agg(tl < 10), "tilt_10-20": agg(tl >= 10)}  # fmt: skip
    tag = (
        ("-noanchor" if a.no_anchor else "")
        + ("-nochecks" if a.no_lock_checks else "")
        + (f"-{a.name}" if a.name else "")
    )
    run = new_run_dir("plate-cameras" + tag)
    for j, (img, _) in enumerate(bad_imgs):
        cv2.imwrite(str(run / f"wrong_{j:02d}.jpg"), img)
    write_metrics(run, m)
    (run / "notes.md").write_text(
        "# Live plate tracker under virtual cameras (pipette1)\n\n"
        f"{len(rows)} samples x {a.frames} frames; scale 0.5-1.6x, any rotation, tilt 0-{a.max_tilt:.0f} deg. "
        "correct = grid within 0.25 spacing (with labels, same A1 rule), from the first lock on.\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in m.items())
        + "\n\nKeep local (third-party video).\n"
    )
    print(m, "->", run)


if __name__ == "__main__":
    main()
