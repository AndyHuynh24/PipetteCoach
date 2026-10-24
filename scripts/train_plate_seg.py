"""Train the plate SEGMENTATION model (YOLO11n-seg, 1 class) for any camera distance / rotation / mild tilt.

Prereq: scripts/build_plate_seg_dataset.py. Output: models/plate_seg/weights/best.pt (gitignored) +
runs/<ts>_plate-seg-train/{metrics.json, notes.md}. YOLO's val = val plate 4 (+ held-out video chunks), labels
from the measured outlines: monitoring; the real checks are scripts/eval_generalization.py and the renders.

    python scripts/train_plate_seg.py [--epochs 50]
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--init", default="yolo11n-seg.pt")
    a = ap.parse_args()
    from ultralytics import YOLO

    out = REPO_ROOT / "models" / "plate_seg"
    if out.exists():
        shutil.rmtree(out)
    YOLO(a.init).train(
        data=str(PROCESSED_DIR / "plate_seg" / "data.yaml"), imgsz=a.imgsz, epochs=a.epochs, batch=8,
        device=a.device, seed=a.seed, deterministic=True, workers=4,
        degrees=180,  # any orientation, upright plates included
        scale=0.9,  # camera distance 0.1-1.9x (plus multi-scale search at run time)
        perspective=0.0005,  # camera not straight above
        fliplr=0.5, flipud=0.5, mosaic=1.0,
        project=str(out.parent), name=out.name, exist_ok=True, plots=False, verbose=False,
    )  # fmt: skip
    rows = list(csv.DictReader(open(out / "results.csv")))
    last = {k.strip(): v for k, v in rows[-1].items()}
    run = new_run_dir("plate-seg-train")
    m = {"epochs": a.epochs, "imgsz": a.imgsz, "seed": a.seed, "init": a.init,
         "val_mask_mAP50_95": float(last["metrics/mAP50-95(M)"]),
         "val_box_mAP50_95": float(last["metrics/mAP50-95(B)"]),
         "weights": str(out / "weights" / "best.pt")}  # fmt: skip
    write_metrics(run, m)
    (run / "notes.md").write_text(
        "# Plate segmentation training\n\nYOLO11n-seg, 1 class, labels = measured plate outlines (dataset clips + "
        "auto-labelled video chunks), upright copies + any-angle / scale / perspective augmentation. "
        f"Val mask mAP50-95 {m['val_mask_mAP50_95']:.3f} (monitoring).\n\nRegenerate: build_plate_seg_dataset.py "
        "(see its flags), then train_plate_seg.py (seed 0).\n"
    )
    print("weights:", out / "weights" / "best.pt", "| run:", run, m)


if __name__ == "__main__":
    main()
