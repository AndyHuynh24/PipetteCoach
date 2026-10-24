"""Dispense-level evaluation of the whole live pipeline (plate grid -> tip -> well -> dwell) on pipette1.mp4.

Reference: configs/p1_dispenses.json (all 32 dispenses of session 1, from the user's tip labels + 3 checked by eye).
Each well a tip stops in is checked once per stop, so this measures what the app reports, unlike the per-frame tip
metrics (those use the same grid for prediction and label, so a wrong grid cancels out there).

Stages (each caches to data/processed/dispense_eval/, so the cheap parts can be re-run alone):
  plate  PlateTracker frame by frame (as live) -> per-frame grid H + outline        (<plate>.npz)
  tips   tip model on the plate crop of every frame -> top-3 tip ends + confidences  (<plate>__<tips>.npz)
  score  replay tip -> well -> dwell, match events to the reference -> runs/<ts>_dispense-eval/

Match: a reported dispense at time t matches a reference stop [t0, t1] if t0 - 0.5 <= t <= t1 + 1.0. Same well =
correct, other well = wrong well; reference stops with no report = missed; reports matching no stop = false.
Time split as the tip model (configs/tip_data.yaml): train < 300 s, val 300-340 s, test >= 340 s.
--ref: another reference (same format; optional keys "rows_toward" for the tracker and "splits" [[name, t0, t1]]).

    python scripts/eval_dispenses.py plate --name base
    python scripts/eval_dispenses.py tips --plate base --model models/tip_all/weights/best.pt --name all
    python scripts/eval_dispenses.py score --plate base --tips all
    python scripts/eval_dispenses.py plate --ref configs/p4_dispenses.json --start 0 --end 147 --name p4
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
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tips import DwellDetector, plate_crop_box, well_of  # noqa: E402

WORK = PROCESSED_DIR / "dispense_eval"
REF = REPO_ROOT / "configs" / "p1_dispenses.json"
SPLITS = [("train", 0, 300), ("val", 300, 340), ("test", 340, 1e9)]


def well_name(w) -> str:
    return f"{'ABCDEFGH'[w[1]]}{w[0] + 1}"


def splits(ref: dict) -> list:
    return [tuple(s) for s in ref["splits"]] if "splits" in ref else SPLITS


def frames(start: float, end: float, ref_path=REF):
    ref = json.loads(Path(ref_path).read_text())
    cap = cv2.VideoCapture(str(REPO_ROOT / ref["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i0, i1 = int(start * fps), int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    for i in range(i0, i1):
        ok, f = cap.read()
        if not ok:
            break
        yield i, i / fps, f


def stage_plate(a) -> None:
    from pipette_coach.localizer.detector import PlateDetector
    from pipette_coach.localizer.live import PlateTracker
    from pipette_coach.localizer.well_net import WellNet

    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    kw = json.loads(a.tracker) if a.tracker else {}
    ref = json.loads(Path(a.ref).read_text())
    tr = PlateTracker(cfg, det, net, **{"verify_every": 30, "rows_toward": ref.get("rows_toward"), **kw})
    fps = cv2.VideoCapture(str(REPO_ROOT / ref["video"])).get(cv2.CAP_PROP_FPS) or 30.0
    n = int((a.end - a.start) * fps) + 5
    hs, outl, ts = np.full((n, 3, 3), np.nan), np.full((n, 4, 2), np.nan, np.float32), np.full(n, np.nan)
    k = 0
    for k, (_, t, f) in enumerate(frames(a.start, a.end, a.ref)):
        o = tr.update(f)
        ts[k] = t
        if o.get("outline") is not None:
            hs[k], outl[k] = o["H"], o["outline"]
        if k % 1500 == 0:
            print(f"plate {t:.0f}s", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(WORK / f"{a.name}.npz", H=hs[: k + 1], outline=outl[: k + 1], t=ts[: k + 1], tracker=json.dumps(kw))
    print("grid share", float(np.isfinite(hs[: k + 1, 0, 0]).mean()), "->", WORK / f"{a.name}.npz")


def stage_tips(a) -> None:
    from ultralytics import YOLO

    from pipette_coach.localizer.detector import _default_device

    p = np.load(WORK / f"{a.plate}.npz")
    model, dev = YOLO(a.model), _default_device()
    n = len(p["t"])
    tips = np.full((n, 3, 3), np.nan, np.float32)  # top-3 (x, y, conf)
    start, end = float(p["t"][0]), float(p["t"][-1]) + 0.01
    for k, (_, _, f) in enumerate(frames(start, end, a.ref)):
        if k >= n:
            break
        if np.isnan(p["outline"][k, 0, 0]):
            continue
        x0, y0, x1, y1 = plate_crop_box(p["outline"][k], 0.25, f.shape[1], f.shape[0])
        r = model.predict(f[y0:y1, x0:x1], imgsz=a.imgsz, conf=0.05, verbose=False, device=dev)[0]
        if len(r.boxes):
            c = r.boxes.conf.cpu().numpy()
            xy = r.keypoints.xy.cpu().numpy()[:, 0] + [x0, y0]
            for j, q in enumerate(np.argsort(-c)[:3]):
                tips[k, j] = [*xy[q], c[q]]
        if k % 1500 == 0:
            print(f"tips {k}/{n}", flush=True)
    np.savez(WORK / f"{a.plate}__{a.name}.npz", tips=tips, model=a.model, imgsz=a.imgsz)
    print("->", WORK / f"{a.plate}__{a.name}.npz")


def replay(H, t, tips, conf: float, dwell_kw: dict) -> list[dict]:
    """Tip -> well -> dwell over cached frames, exactly as live (most confident tip >= conf)."""
    dwell, events = DwellDetector(**dwell_kw), []
    for k in range(len(t)):
        well = None
        if np.isfinite(H[k, 0, 0]) and np.isfinite(tips[k, 0, 2]) and tips[k, 0, 2] >= conf:
            well = well_of(H[k], tips[k, 0, :2])
        ev = dwell.update(float(t[k]), well)
        if ev is not None:
            e = {"t": round(float(t[k]), 2), "well": well_name(ev)}
            if dwell.revised:  # longer stop in the same trip: replaces the reported well
                events[-1] = {**e, "t": events[-1]["t"], "revised_from": events[-1]["well"]}
            else:
                events.append(e)
    return events


def match(events: list[dict], ref: list[dict]) -> dict:
    used, rows = set(), []
    for r in ref:
        cand = [i for i, e in enumerate(events) if i not in used and r["t0"] - 0.5 <= e["t"] <= r["t1"] + 1.0]
        hit = [i for i in cand if events[i]["well"] == r["well"]]
        if hit:
            used.add(hit[0])
            rows.append({**r, "result": "correct", "got": events[hit[0]]})
        elif cand:
            used.add(cand[0])
            rows.append({**r, "result": "wrong well", "got": events[cand[0]]})
        else:
            rows.append({**r, "result": "missed"})
    false = [e for i, e in enumerate(events) if i not in used]
    return {"rows": rows, "false": false}


def summary(m: dict, lo: float, hi: float) -> dict:
    rows = [r for r in m["rows"] if lo <= r["t0"] < hi]
    false = [e for e in m["false"] if lo <= e["t"] < hi]
    res = [r["result"] for r in rows]
    return {"stops": len(rows), "correct": res.count("correct"), "wrong_well": res.count("wrong well"),
            "missed": res.count("missed"), "false": len(false)}  # fmt: skip


def grid_off(H: np.ndarray, half: int = 150, step: int = 3) -> tuple[int, int]:
    """Frames (every step-th) whose grid is > 0.5 spacing from the median grid of the frames within +-half (the
    plate sits still; slow camera drift moves the median too): a short wrong stretch (partial view read as other
    rows / columns) stands out. Returns (frames off, frames checked)."""
    from pipette_coach.localizer.geometry import GRID, pitch_px, project

    ok = np.isfinite(H[:, 0, 0])
    pts = np.full((len(H), len(GRID), 2), np.nan)
    pts[ok] = [project(h, GRID) for h in H[ok]]
    off = n = 0
    for k in range(0, len(H), step):
        if not ok[k]:
            continue
        med = np.nanmedian(pts[max(0, k - half) : k + half + 1], axis=0)
        n += 1
        off += np.linalg.norm(pts[k] - med, axis=1).mean() / pitch_px(H[k]) > 0.5
    return int(off), n


def stage_score(a) -> dict:
    ref = json.loads(Path(a.ref).read_text())
    p = np.load(WORK / f"{a.plate}.npz")
    tp = np.load(WORK / f"{a.plate}__{a.tips}.npz")
    lo, hi = ref["covers"]
    keep = (p["t"] >= lo) & (p["t"] < hi)
    dwell_kw = json.loads(a.dwell) if a.dwell else {}
    events = replay(p["H"][keep], p["t"][keep], tp["tips"][keep], a.conf, dwell_kw)
    m = match(events, ref["events"])
    out = {"plate": a.plate, "tips": a.tips, "model": str(tp["model"]), "tracker": str(p["tracker"]), "conf": a.conf,
           "dwell": dwell_kw, "grid_share": float(np.isfinite(p["H"][keep, 0, 0]).mean()),
           "grid_off_frames": grid_off(p["H"][keep]),
           "all": summary(m, 0, 1e9), **{s: summary(m, lo_, hi_) for s, lo_, hi_ in splits(ref)}}  # fmt: skip
    if not a.quiet:
        run = new_run_dir(f"dispense-eval-{a.plate}-{a.tips}")
        write_metrics(run, out)
        (run / "events.json").write_text(json.dumps(m, indent=1))
        bad = [f"{r['well']} {r['t0']}-{r['t1']}s: {r['result']} {r.get('got', '')}" for r in m["rows"]
               if r["result"] != "correct"]  # fmt: skip
        bad += [f"false {e}" for e in m["false"]]
        (run / "notes.md").write_text(
            f"# Dispense eval ({a.plate} / {a.tips})\n\nReference {a.ref} ({len(ref['events'])} stops). "
            f"Summary: {out['all']}\n\nNot correct:\n" + "\n".join(f"- {b}" for b in bad) + "\n"
        )
        print(json.dumps(out, indent=1), "\n", "\n".join(bad), "\n->", run)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plate", "tips", "score"])
    ap.add_argument("--ref", default=str(REF), help="reference dispenses (default: pipette1 session 1)")
    ap.add_argument("--name", default="base")
    ap.add_argument("--start", type=float, default=80.0)
    ap.add_argument("--end", type=float, default=390.0)
    ap.add_argument("--tracker", default="", help='PlateTracker kwargs as JSON, e.g. {"track_every": 2}')
    ap.add_argument("--plate", default="base")
    ap.add_argument("--model", default="models/tip_all/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--tips", default="all")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--dwell", default="", help='DwellDetector kwargs as JSON, e.g. {"min_s": 0.3}')
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    {"plate": stage_plate, "tips": stage_tips, "score": stage_score}[a.stage](a)


if __name__ == "__main__":
    main()
