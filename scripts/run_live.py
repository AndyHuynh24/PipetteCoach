"""Run the real-time PlateTracker frame by frame (as a live camera would feed it) on a video or dataset clips.

Measures: tracker time per frame (p50 / p95 / mean, frames per second), share of frames with a grid; on dataset
clips also accuracy against the approved reference grids (data/processed/plate_grids/, the edges + wells method
checked by eye): placement error in well spacings (labelling-free) and A1 orientation agreement.
Output: runs/<ts>_live/{live.mp4, metrics.json, notes.md}. Test plates and excluded plates (plate 10) are refused.

    python scripts/run_live.py --video data/raw/pipette1.mp4 --seconds 60
    python scripts/run_live.py --plates 4,1,2,9
"""

import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_all import encode  # noqa: E402
from render_plate_rebuild import draw  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import grid_error  # noqa: E402
from pipette_coach.localizer.live import PlateTracker  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

ZOOM_W, ZOOM_H = 640, 460


def add_zoom(img: np.ndarray, out: dict) -> np.ndarray:
    """Zoom-in view of the plate (grid outline, or else the detector box) in the top-right corner."""
    if out.get("outline") is not None:
        q = np.asarray(out["outline"])
    elif out.get("box") is not None:
        b = out["box"]
        q = np.array([[b[0], b[1]], [b[2], b[3]]])
    else:
        return img
    h, w = img.shape[:2]
    c, ext = q.mean(0), np.ptp(q, axis=0).max() * 1.25
    cw = int(max(ext, 60))
    ch = int(cw * ZOOM_H / ZOOM_W)
    x0 = int(np.clip(c[0] - cw / 2, 0, max(w - cw, 0)))
    y0 = int(np.clip(c[1] - ch / 2, 34, max(h - ch, 34)))
    crop = img[y0 : y0 + ch, x0 : x0 + cw]
    if crop.size == 0:
        return img
    z = cv2.resize(crop, (ZOOM_W, ZOOM_H), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(z, (0, 0), (ZOOM_W - 1, ZOOM_H - 1), (255, 255, 255), 3)
    img[44 : 44 + ZOOM_H, w - ZOOM_W - 10 : w - 10] = z
    return img


def draw_log(img: np.ndarray, events: list[dict]) -> None:
    """Dispense log (last 8) top-left. With a well-change verdict (event["liquid"]): True = liquid seen (green),
    None = not seen clearly (orange), no key yet = waiting (grey); False = the well did not change (a hover, not a
    dispense): left out of the log."""
    log = [e for e in events if e.get("liquid") is not False][-8:]
    n = sum(e.get("liquid") is not False for e in events)
    cv2.rectangle(img, (10, 44), (400, 80 + 34 * len(log)), (0, 0, 0), -1)
    cv2.putText(img, f"dispenses: {n}", (20, 72), 0, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    tags = {True: (" liquid", (0, 220, 0)), False: (" no liquid", (0, 0, 255)), None: (" ?", (0, 200, 255))}
    for j, e in enumerate(log):
        tag, col = tags[e["liquid"]] if "liquid" in e else ("", (200, 200, 200))
        cv2.putText(img, f"{e['t']:6.1f}s  {e['well']}{tag}", (20, 106 + 34 * j), 0, 0.8, col, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--plates", default=None, help="dataset plates (train/val only), e.g. 4,1,2,9")
    ap.add_argument("--clips", default="")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--detector", default="models/plate_det/weights/best.pt")
    ap.add_argument("--no-video", action="store_true", help="metrics only (no render)")
    ap.add_argument("--name", default="live")
    ap.add_argument("--track-every", type=int, default=1, help="track the plate every n-th frame (live: 2)")
    ap.add_argument("--no-anchor", action="store_true", help="tracker without the still-plate memory (comparison)")
    ap.add_argument("--verify-every", type=int, default=60)
    a = ap.parse_args()
    cfg = load_config("localizer")
    cfg["detector"]["weights"] = a.detector
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    run = new_run_dir(a.name)
    vw, ms, statuses, errs, a1_ok, n_ref = None, [], collections.Counter(), [], 0, 0
    per_clip: dict = {}

    def emit(frame, out, title):
        nonlocal vw
        if a.no_video:
            return
        img = draw(frame, out["box"], out if out["H"] is not None else None, out.get("wells"),
                   f"{title}{out['status']}   {out['ms']:.0f} ms")  # fmt: skip
        img = add_zoom(img, out)
        if vw is None:
            vw = cv2.VideoWriter(str(run / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, img.shape[1::-1])
        vw.write(img)

    # warm-up (first model calls compile kernels)
    tr0 = PlateTracker(cfg, det, net)
    tr0.update(np.zeros((1080, 1920, 3), np.uint8))
    t_start = time.time()
    if a.plates:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cache_plate_grids import load_grid_windows

        from pipette_coach.data import clips_for_split, load_splits, plate_of, plate_rows_toward
        from pipette_coach.localizer.plate_grid import window_at
        from pipette_coach.localizer.session import read_frames

        splits = load_splits()  # excluded plates (plate 10) are already dropped here
        plates = [int(x) for x in a.plates.split(",") if x]
        allowed = set(splits["train"] + splits["val"])
        if any(p not in allowed for p in plates):
            sys.exit("only train/val plates (test and excluded plates are refused)")
        only = {c for c in a.clips.split(",") if c}
        for pl in plates:
            for clip in [c for sp in ("train", "val") for c in clips_for_split(sp) if plate_of(c) == pl]:
                if only and clip not in only:
                    continue
                tracker = PlateTracker(cfg, det, net, rows_toward=plate_rows_toward(pl), track_every=a.track_every,
                                       anchor=not a.no_anchor, verify_every=a.verify_every)  # fmt: skip
                e0, a0, r0 = len(errs), a1_ok, n_ref
                wins = load_grid_windows(clip)
                for i, f in enumerate(read_frames(clip)):
                    out = tracker.update(f)
                    ms.append(out["ms"])
                    statuses[out["status"].split(" (")[0].split(" [")[0]] += 1
                    w = window_at(wins, i) if wins else None
                    if w is not None and w["H_grid"] is not None:
                        n_ref += 1
                        if out["H"] is not None:
                            errs.append(grid_error(out["H"], w["H_grid"]))
                            a1_ok += grid_error(out["H"], w["H_grid"], ignore_labelling=False) < 0.5
                    emit(f, out, f"{clip}  ")
                ce = np.array(errs[e0:])
                per_clip[clip] = {"frames_with_ref": n_ref - r0, "grid": len(ce),
                                  "within_0.25": int((ce < 0.25).sum()), "a1_ok": a1_ok - a0,
                                  "median_err": float(np.median(ce)) if len(ce) else None}  # fmt: skip
                print(f"{clip} {per_clip[clip]} ({time.time() - t_start:.0f}s)", flush=True)
        src = f"plates {a.plates}"
    else:
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        tracker = PlateTracker(cfg, det, net, track_every=a.track_every, anchor=not a.no_anchor,
                               verify_every=a.verify_every)  # fmt: skip
        for i in range(int(a.seconds * fps) if a.seconds > 0 else 10**9):
            ok, f = cap.read()
            if not ok:
                break
            out = tracker.update(f)
            ms.append(out["ms"])
            statuses[out["status"].split(" (")[0].split(" [")[0]] += 1
            emit(f, out, f"t={i / fps:5.1f}s  ")
        src = a.video
    if vw is not None:
        vw.release()
        encode(run / "raw.mp4", run / "live.mp4")
    ms = np.array(ms)
    n = len(ms)
    grid = sum(v for k, v in statuses.items() if k.startswith(("tracked", "rebuilt", "held", "re-found")))
    m = {"source": src, "track_every": a.track_every, "anchor": not a.no_anchor, "verify_every": a.verify_every,
         "frames": n,
         "ms_p50": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)),
         "ms_mean": float(ms.mean()), "fps_mean": float(1000 / ms.mean()), "grid_share": grid / max(n, 1),
         "status": dict(statuses), "per_clip": per_clip}  # fmt: skip
    if n_ref:
        e = np.array(errs)
        m |= {"ref_frames": n_ref, "grid_on_ref_frames": len(e) / n_ref,
              "within_0.25_pitch_of_reference": float((e < 0.25).mean()) if len(e) else None,
              "median_error_pitch": float(np.median(e)) if len(e) else None,
              "a1_agrees_with_reference": a1_ok / max(len(e), 1)}  # fmt: skip
    write_metrics(run, m)
    (run / "notes.md").write_text(f"# Live tracker ({src})\n\n{m}\n\nTracker time excludes drawing. Keep local.\n")
    print(m)


if __name__ == "__main__":
    main()
