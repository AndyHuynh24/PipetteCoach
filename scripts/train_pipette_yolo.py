"""Fine-tune YOLO11n to detect the pipette and its tip on plate crops (2 classes, as in the KicoSVK YOLOv7 project).

Prereq: scripts/build_pipette_yolo_dataset.py (pseudo-labels; hand-clicked tips override when present).
Output: models/pipette_yolo/weights/best.pt (gitignored) + runs/<ts>_pipette-yolo-train/{metrics.json, notes.md}.
To fine-tune further after labelling more tips: rebuild the dataset, then rerun this (it starts from the COCO
nano weights each time; pass --init models/pipette_yolo/weights/best.pt to continue from the last model).

    python scripts/train_pipette_yolo.py [--epochs 80] [--imgsz 640]
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
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--init", default="yolo11n.pt", help="starting weights (COCO nano, or a previous best.pt)")
    a = ap.parse_args()

    from ultralytics import YOLO

    out = REPO_ROOT / "models" / "pipette_yolo"
    if out.exists() and a.init == "yolo11n.pt":
        shutil.rmtree(out)
    model = YOLO(a.init)
    model.train(
        data=str(PROCESSED_DIR / "pipette_yolo" / "data.yaml"),
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=16,
        device=a.device,
        seed=a.seed,
        deterministic=True,
        workers=4,
        degrees=10,
        fliplr=0.5,  # the pipette comes in from any side of the plate
        flipud=0.5,
        mosaic=1.0,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    rows = list(csv.DictReader(open(out / "results.csv")))
    last = {k.strip(): v for k, v in rows[-1].items()}
    run = new_run_dir("pipette-yolo-train")
    m = {"epochs": a.epochs, "imgsz": a.imgsz, "seed": a.seed, "init": a.init,
         "yolo_val": "held-out TRAIN clips, pseudo-labels (monitoring only)",
         "val_mAP50": float(last["metrics/mAP50(B)"]), "val_mAP50_95": float(last["metrics/mAP50-95(B)"]),
         "weights": str(out / "weights" / "best.pt")}  # fmt: skip
    write_metrics(run, m)
    (run / "notes.md").write_text(
        "# Pipette + tip YOLO\n\nYOLO11n detect, classes pipette / tip, plate crops, pseudo-labels from the plate-view "
        f"foreground (+ hand-clicked tips if any). YOLO val mAP50 {m['val_mAP50']:.3f} on held-out TRAIN clips with "
        "pseudo-labels: monitoring only, not a real accuracy. Real check: scripts/render_pipette_yolo.py on val "
        "plate 4.\n\nRegenerate: build_pipette_yolo_dataset.py, then train_pipette_yolo.py (seed 0).\n"
    )
    print("weights:", out / "weights" / "best.pt", "| run:", run, "| mAP50", m["val_mAP50"])


if __name__ == "__main__":
    main()
