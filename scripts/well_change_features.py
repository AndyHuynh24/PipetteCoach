"""Per-frame inputs of the old liquid check (pipette_coach/well_change.py) for one of our recordings, as live: the
whole video, frame by frame, PlateTracker grid (new well_net), the tip (TipFinder, as live_demo) and the per-well
features the check compares (texture, Lab inside the well, Lab of the cell corners).
The check looks up to 5 s before the tip came and 4 s after it left, so it needs the continuous video (not only the
SPACE clips). Used by scripts/tune_well_change_live.py.

Output data/processed/well_change/<rec>.npz: t (n,), H (n, 3, 3) (NaN = no grid), tip (n, 3) x, y, conf (NaN = none),
F (n, 8, 12, 7) (NaN = no grid).

    python scripts/well_change_features.py --rec pipette8
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, WELL_NET, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.well_change import cell_features, well_cells  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="pipette8")
    ap.add_argument("--tip-model", default="models/tip_colors/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=480, help="tip model input size (live_demo's default)")
    a = ap.parse_args()
    from ultralytics import YOLO

    from pipette_coach.live_pipeline import TipFinder
    from pipette_coach.localizer.detector import PlateDetector
    from pipette_coach.localizer.live import PlateTracker
    from pipette_coach.localizer.well_net import WellNet

    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    tracker = PlateTracker(cfg, PlateDetector(cfg), WellNet(REPO_ROOT / WELL_NET))
    tip = TipFinder(YOLO(str(REPO_ROOT / a.tip_model)), imgsz=a.imgsz)
    cap = cv2.VideoCapture(str(REPO_ROOT / "data" / "own" / a.rec / "video.mp4"))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T, Hs, tips = np.arange(n) / fps, np.full((n, 3, 3), np.nan), np.full((n, 3), np.nan)
    F = np.full((n, 8, 12, 7), np.nan, np.float32)
    t0 = time.time()
    for k in range(n):
        ok, frame = cap.read()
        if not ok:
            T, Hs, tips, F = T[:k], Hs[:k], tips[:k], F[:k]
            break
        p = tracker.update(frame)
        if p.get("H") is None:
            continue
        Hs[k] = p["H"]
        F[k] = cell_features(well_cells(frame, p["H"]))
        xy, _ = tip(frame, p)
        if xy is not None:
            tips[k] = xy[0], xy[1], 1.0
        if k % 900 == 0:
            print(f"{k}/{n} frames, {k / max(time.time() - t0, 1e-6):.1f} fps", flush=True)
    out = PROCESSED_DIR / "well_change"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.rec}.npz", t=T, H=Hs, tip=tips, F=F)
    print(f"{len(T)} frames -> {out / f'{a.rec}.npz'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
