"""Real-time budget: plate tracker + a tip model, per frame, as a live camera would feed them.

The tip model is not trained yet; its SPEED depends only on the architecture, so the stock YOLO11n-pose
(COCO weights) is timed on the same input the real tip model will get: the plate area + margin (from the
tracker's outline) resized to --tip-imgsz. Reports per-stage and total time per frame and frames per second, for
the tip model on every frame and on every 2nd frame. Frames are read before timing (camera capture excluded).

    python scripts/bench_realtime.py --video data/raw/pipette1.mp4 --seconds 60
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector, _default_device  # noqa: E402
from pipette_coach.localizer.live import PlateTracker  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402


def plate_crop(frame: np.ndarray, out: dict, margin: float = 0.25) -> np.ndarray | None:
    q = out.get("outline")
    if q is None and out.get("box") is not None:
        b = out["box"]
        q = np.array([[b[0], b[1]], [b[2], b[3]]])
    if q is None:
        return None
    lo, hi = np.asarray(q).min(0), np.asarray(q).max(0)
    m = (hi - lo) * margin
    x0, y0 = np.maximum(lo - m, 0).astype(int)
    x1, y1 = np.minimum(hi + m, frame.shape[1::-1]).astype(int)
    return frame[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--tip-imgsz", type=int, default=640)
    ap.add_argument("--track-every", type=int, default=1)
    ap.add_argument("--tip-every", default="1,2")
    a = ap.parse_args()
    from ultralytics import YOLO

    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    tip = YOLO("yolo11n-pose.pt")  # stand-in with the same architecture as the future tip model
    dev = _default_device()
    res = {}
    for every in [int(x) for x in a.tip_every.split(",")]:
        # frames are read one at a time (like a camera) and the read is NOT timed
        cap = cv2.VideoCapture(a.video)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(a.seconds * fps_in)
        tracker = PlateTracker(cfg, det, net, track_every=a.track_every)
        ok, f0 = cap.read()
        tracker.update(f0)  # warm-up
        tip.predict(f0[:640, :640], imgsz=a.tip_imgsz, verbose=False, device=dev)
        t_track, t_tip, t_total = [], [], []
        for i in range(n):
            ok, f = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            out = tracker.update(f)
            t1 = time.perf_counter()
            if i % every == 0:
                crop = plate_crop(f, out)
                if crop is not None:
                    tip.predict(crop, imgsz=a.tip_imgsz, verbose=False, device=dev)
            t2 = time.perf_counter()
            t_track.append((t1 - t0) * 1000)
            t_tip.append((t2 - t1) * 1000)
            t_total.append((t2 - t0) * 1000)
        tt = np.array(t_total)
        res[f"tip_every_{every}"] = {
            "tracker_ms_median": float(np.median(t_track)), "tip_ms_median_when_run": float(np.median(
                [x for k, x in enumerate(t_tip) if k % every == 0])),
            "total_ms_median": float(np.median(tt)), "total_ms_p95": float(np.percentile(tt, 95)),
            "fps_mean": float(1000 / tt.mean()), "frames_over_33ms": float((tt > 1000 / 30).mean()),
        }  # fmt: skip
        print(every, res[f"tip_every_{every}"], flush=True)
    run = new_run_dir("bench-realtime")
    write_metrics(run, {"video": a.video, "seconds": a.seconds, "device": dev, "tip_imgsz": a.tip_imgsz,
                        "track_every": a.track_every, **res})  # fmt: skip
    note = "Tip model = stock YOLO11n-pose (same architecture as the future tip model), on the plate crop."
    (run / "notes.md").write_text(f"# Real-time budget\n\n{res}\n\n{note} Camera capture not included.\n")


if __name__ == "__main__":
    main()
