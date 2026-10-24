"""Pick frames for tip annotation: moments when something (pipette / hand) is over the plate.

Every --step seconds: the plate detector (multi-scale) gives the plate box; the share of pixels inside the box
that differ from the plate's clean look (running median of recent plate crops) measures activity over the plate.
Frames with activity in [lo, hi] are candidates (a pipette, not an empty plate or an arm covering everything);
--n of them are kept, spread evenly over the video, plus a few quiet frames as "no pipette" examples. File names
carry the time (do not rename: the train / test split is by time).

    python scripts/extract_tip_frames.py --video data/raw/pipette1.mp4 --n 400
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_plate_det import detect  # noqa: E402

from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--quiet", type=int, default=30, help="extra frames with nothing over the plate")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.03)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--detector", default="models/plate_det/weights/best.pt")
    a = ap.parse_args()
    cfg = load_config("localizer")
    cfg["detector"]["weights"] = a.detector
    det = PlateDetector(cfg)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hist: list[np.ndarray] = []
    busy, quiet = [], []
    for idx in range(0, n_frames, max(1, int(a.step * fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            break
        b = detect(det, f)
        if not len(b):
            continue
        x0, y0, x1, y1 = b[0].astype(int)
        crop = cv2.resize(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (96, 64)).astype(np.float32)
        if len(hist) >= 5:
            ref = np.median(np.stack(hist[-20:]), axis=0)
            act = float((np.abs(crop - ref) > 25).mean())
            (busy if a.lo <= act <= a.hi else quiet if act < 0.01 else []).append(idx)
        hist.append(crop)
    rng = np.random.default_rng(0)
    pick = sorted(np.array(busy)[np.linspace(0, len(busy) - 1, min(a.n, len(busy))).astype(int)].tolist())
    pick += sorted(rng.choice(quiet, min(a.quiet, len(quiet)), replace=False).tolist()) if quiet else []
    out = PROCESSED_DIR / "annotation_frames" / f"tips_{Path(a.video).stem}"
    out.mkdir(parents=True, exist_ok=True)
    for idx in sorted(set(pick)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            cv2.imwrite(str(out / f"{Path(a.video).stem}_t{idx / fps:06.1f}s_f{idx:05d}.jpg"), f,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])  # fmt: skip
    print(f"{len(busy)} busy / {len(quiet)} quiet candidates; wrote {len(set(pick))} frames -> {out}")


if __name__ == "__main__":
    main()
