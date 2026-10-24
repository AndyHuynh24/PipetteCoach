"""Train the YOLO11n oriented-box plate detector on auto-labelled frames.

Prereqs: scripts/autolabel_plates.py (train + val), scripts/build_plate_yolo_dataset.py
Output: models/plate_obb/weights/best.pt (gitignored) + runs/<ts>_plate-detector-train/{metrics.json,notes.md}
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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--data", default="plate_yolo", help="dataset folder in data/processed/")
    ap.add_argument("--out", default="plate_obb", help="model folder in models/")
    ap.add_argument(
        "--init", default="yolo11n-obb.pt", help="starting weights (COCO/DOTA nano OBB or a previous best.pt)"
    )
    ap.add_argument("--degrees", type=float, default=20, help="rotation augmentation (+-deg); 180 = any orientation")
    ap.add_argument("--scale", type=float, default=0.5, help="scale augmentation gain (1 +- scale)")
    ap.add_argument("--perspective", type=float, default=0.0, help="perspective augmentation (camera tilt)")
    a = ap.parse_args()

    from ultralytics import YOLO

    out = REPO_ROOT / "models" / a.out
    if out.exists():
        shutil.rmtree(out)
    model = YOLO(a.init)
    model.train(
        data=str(PROCESSED_DIR / a.data / "data.yaml"),
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=8,
        device=a.device,
        seed=a.seed,
        deterministic=True,
        workers=4,
        degrees=a.degrees,  # plates appear at any rotation on the bench (v3: 180 - upright plates too)
        scale=a.scale,  # other camera distances (v3: 0.9 -> 0.1-1.9x)
        perspective=a.perspective,  # camera not straight above the plate
        fliplr=0.5,
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
    run = new_run_dir(f"{a.out.replace('_', '-')}-train")
    write_metrics(
        run,
        {
            "epochs": a.epochs,
            "init": a.init,
            "degrees": a.degrees,
            "scale": a.scale,
            "perspective": a.perspective,
            "imgsz": a.imgsz,
            "seed": a.seed,
            "val_split": "plate 4 (auto-labels)",
            "train_data": a.data,
            "val_mAP50": float(last["metrics/mAP50(B)"]),
            "val_mAP50_95": float(last["metrics/mAP50-95(B)"]),
            "weights": str(out / "weights" / "best.pt"),
        },
    )
    (run / "notes.md").write_text(
        "# Plate detector training\n\n"
        "YOLO11n-OBB, 1 class (plate), trained on frames auto-labelled by the global grid fit "
        "(train plates 1, 2, 9, 10; only fits with >= 90% of wells on rings). Val = plate 4 auto-labels.\n\n"
        f"Result: val mAP50-95 {float(last['metrics/mAP50-95(B)']):.3f}. Note the val labels come from the grid "
        "fit, not humans; see the localizer eval run for grid accuracy.\n\n"
        "Regenerate: `python scripts/autolabel_plates.py --split train` and `--split val`, "
        "`python scripts/build_plate_yolo_dataset.py`, `python scripts/train_plate_detector.py`.\n"
    )
    print("weights:", out / "weights" / "best.pt", "| run:", run)


if __name__ == "__main__":
    main()
