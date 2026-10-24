"""Score plate detectors on the hand-labelled HELD-OUT frames (data/processed/plate_det_test, never trained on).

Per frame, the detector runs with the multi-scale search (first scale whose best box passes detector.min_conf).
Metrics: found = best box overlaps the labelled (visible) plate with IoU >= 0.5; mean IoU; false detections =
accepted boxes that overlap no labelled plate (IoU < 0.1); split = two or more accepted boxes inside the same
labelled plate (one plate seen as two). Old (OBB) models are scored by the axis-aligned box of their quad.

    python scripts/eval_plate_det.py --models models/plate_obb_v2/weights/best.pt,models/plate_det/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector, boxes_of  # noqa: E402
from pipette_coach.localizer.plate_grid import SEARCH_SCALES  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def detect(det: PlateDetector, img: np.ndarray) -> np.ndarray:
    """Accepted boxes (x0, y0, x1, y1) at the first search scale that finds the plate."""
    for s0 in SEARCH_SCALES:
        g = img if s0 == 1.0 else cv2.resize(img, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA if s0 < 1
                                             else cv2.INTER_LINEAR)  # fmt: skip
        r = det.model.predict(g, imgsz=det.imgsz, conf=det.min_conf, verbose=False, device=det.device)[0]
        q, c = boxes_of(r)
        if len(c):
            q = q / s0
            return np.concatenate([q.min(1), q.max(1)], axis=1)
    return np.zeros((0, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/plate_obb_v2/weights/best.pt,models/plate_det/weights/best.pt")
    a = ap.parse_args()
    test = PROCESSED_DIR / "plate_det_test"
    items = []
    for p in sorted((test / "images" / "test").glob("*.jpg")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gts = []
        for line in (test / "labels" / "test" / f"{p.stem}.txt").read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            gts.append(np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
        items.append((p.stem, img, gts))
    lines = ["# Plate detectors on held-out hand-labelled frames", "", f"{len(items)} frames (video 300-516 s, incl. "
             "the camera change at 390 s). Visible-plate boxes.", "", "| model | found (IoU>=0.5) | mean IoU | "
             "false detections | split plate |", "|---|---|---|---|---|"]  # fmt: skip
    metrics = {}
    for mp in [m for m in a.models.split(",") if m]:
        cfg = load_config("localizer")
        cfg["detector"]["weights"] = mp
        det = PlateDetector(cfg)
        found, ious, fp, split, misses = 0, [], 0, 0, []
        for name, img, gts in items:
            boxes = detect(det, img)
            for g in gts:
                best = max((iou(b, g) for b in boxes), default=0.0)
                ious.append(best)
                found += best >= 0.5
                if best < 0.5:
                    misses.append(name)
                inside = [b for b in boxes if iou(b, g) > 0.05 and
                          ((np.minimum(b[2:], g[2:]) - np.maximum(b[:2], g[:2])).clip(0).prod()
                           > 0.6 * (b[2:] - b[:2]).prod())]  # fmt: skip
                split += len(inside) >= 2
            fp += sum(all(iou(b, g) < 0.1 for g in gts) for b in boxes)
        n = sum(len(g) for _, _, g in items)
        metrics[mp] = {"found": found / n, "mean_iou": float(np.mean(ious)), "false_detections": fp, "split": split,
                       "missed_frames": misses}  # fmt: skip
        lines.append(f"| {mp} | {found}/{n} | {np.mean(ious):.2f} | {fp} | {split} |")
        print(lines[-1], "missed:", misses, flush=True)
    run = new_run_dir("plate-det-eval")
    write_metrics(run, metrics)
    (run / "notes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
