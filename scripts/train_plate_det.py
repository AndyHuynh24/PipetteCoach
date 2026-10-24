"""Train the plate DETECTOR (YOLO11n detect, 1 class "plate", visible plate) for any camera / pose.

Prereq: scripts/build_plate_det_dataset.py. Output: models/plate_det/weights/best.pt (gitignored) +
runs/<ts>_plate-det-train/{metrics.json, notes.md}. Checkpoint choice on val plate 4; the hand-labelled held-out
video frames (data/processed/plate_det_test) are scored separately by scripts/eval_plate_det.py.

    python scripts/train_plate_det.py [--epochs 60]
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
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--init", default="yolo11n.pt", help="or a trained plate_det .pt to fine-tune")
    ap.add_argument("--name", default="plate_det", help="models/<name> (plate_det = the one the live app loads)")
    a = ap.parse_args()
    from ultralytics import YOLO

    out = REPO_ROOT / "models" / a.name
    if out.exists():
        shutil.rmtree(out)
    YOLO(a.init).train(
        data=str(PROCESSED_DIR / "plate_det" / "data.yaml"), imgsz=a.imgsz, epochs=a.epochs, batch=8,
        device=a.device, seed=a.seed, deterministic=True, workers=4,
        degrees=15,  # boxes are axis-aligned: large rotations come from the upright copies instead
        scale=0.9,  # camera distance 0.1-1.9x (plus multi-scale search at run time)
        perspective=0.0005, fliplr=0.5, flipud=0.5, mosaic=1.0,
        project=str(out.parent), name=out.name, exist_ok=True, plots=False, verbose=False,
    )  # fmt: skip
    rows = list(csv.DictReader(open(out / "results.csv")))
    last = {k.strip(): v for k, v in rows[-1].items()}
    run = new_run_dir("plate-det-train")
    m = {"epochs": a.epochs, "imgsz": a.imgsz, "seed": a.seed, "init": a.init, "name": a.name,
         "val_mAP50": float(last["metrics/mAP50(B)"]), "val_mAP50_95": float(last["metrics/mAP50-95(B)"]),
         "weights": str(out / "weights" / "best.pt")}  # fmt: skip
    write_metrics(run, m)
    (run / "notes.md").write_text(
        "# Plate detector (visible plate)\n\nYOLO11n detect, 1 class. Train: hand-labelled video frames before 5:00 "
        "(+ upright copies), dataset plates (boxes from the approved edges + wells method, + upright copies), "
        "synthetic tilted/lifted plates. Checkpoint choice: val plate 4. Held-out test: scripts/eval_plate_det.py.\n"
        "\nRegenerate: build_plate_seg_dataset.py, build_plate_synth.py, build_plate_det_dataset.py, then this.\n"
    )
    print("weights:", out / "weights" / "best.pt", "| run:", run, m)


if __name__ == "__main__":
    main()
