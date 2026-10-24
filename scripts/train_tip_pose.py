"""Train the tip-end model: YOLO11n-pose, 1 class "tip", 1 keypoint (the tip end), on plate crops / tip crops.

Prereq: scripts/build_tip_dataset.py build --name <data>. Output: models/tip_<name>/weights/best.pt (gitignored)
+ runs/<ts>_tip-train-<name>/{metrics.json, notes.md}. Checkpoint choice on the val split only.

    python scripts/train_tip_pose.py --data tips_all --name all
    python scripts/train_tip_pose.py --data tips_all --name final --full --epochs 70   # final model, all labels

--full: train on train + val + test (after the val/test numbers are recorded). Nothing is held out, so the
epoch count is fixed (from the plateau of the held-out run) and last.pt is the model; the val metrics it logs
are on frames it trained on (no meaning).
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tips_all")
    ap.add_argument("--name", default="all")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--init", default="yolo11n-pose.pt")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from its last.pt")
    ap.add_argument("--full", action="store_true", help="train on all splits, fixed epochs, use last.pt")
    ap.add_argument("--hold-out", default="", help="leave this video out of train and val (new-scene test)")
    ap.add_argument("--aug", default="", help="training kwargs overriding the defaults, as JSON")
    a = ap.parse_args()
    from ultralytics import YOLO

    out = REPO_ROOT / "models" / f"tip_{a.name}"
    if a.resume:
        YOLO(str(out / "weights" / "last.pt")).train(resume=True)
    else:
        if out.exists():
            shutil.rmtree(out)
        train(YOLO(a.init), a, out)
    report(a, out)


def train(model, a, out: Path) -> None:
    data = PROCESSED_DIR / f"pose_{a.data}" / "data.yaml"
    if a.full:
        full = data.with_name("data_full.yaml")
        all_splits = "train: [images/train, images/val, images/test]"
        full.write_text(data.read_text().replace("train: images/train", all_splits))
        data = full
    if a.hold_out:  # image lists without that video (labels are found next to the images, as usual)
        root = data.parent
        for sp in ("train", "val"):
            files = [str(f) for f in sorted((root / "images" / sp).glob("*.jpg")) if not f.name.startswith(a.hold_out)]
            (root / f"lo_{a.hold_out}_{sp}.txt").write_text("\n".join(files) + "\n")
        lo = root / f"data_lo_{a.hold_out}.yaml"
        lo.write_text(data.read_text().replace("train: images/train", f"train: lo_{a.hold_out}_train.txt")
                      .replace("val: images/val", f"val: lo_{a.hold_out}_val.txt"))  # fmt: skip
        data = lo
    kw = dict(
        degrees=30, fliplr=0.5, flipud=0.5,  # camera can face the plate from any side: flips + tilt
        scale=0.5, translate=0.1, mosaic=1.0, close_mosaic=10, hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
    )  # fmt: skip
    kw.update(json.loads(a.aug) if a.aug else {})
    model.train(
        data=str(data), imgsz=a.imgsz, epochs=a.epochs, batch=16,
        patience=0 if a.full else 30, device=a.device, seed=a.seed, deterministic=True, workers=4,
        project=str(out.parent), name=out.name, exist_ok=True, plots=False, verbose=False, **kw,
    )  # fmt: skip


def report(a, out: Path) -> None:
    rows = list(csv.DictReader(open(out / "results.csv")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(P)"]))
    run = new_run_dir(f"tip-train-{a.name}")
    m = {"data": a.data, "epochs_run": len(rows), "imgsz": a.imgsz, "seed": a.seed, "init": a.init,
         "best_epoch": int(float(best["epoch"])), "val_box_mAP50": float(best["metrics/mAP50(B)"]),
         "val_kpt_mAP50": float(best["metrics/mAP50(P)"]), "val_kpt_mAP50_95": float(best["metrics/mAP50-95(P)"]),
         "full": a.full, "hold_out": a.hold_out, "aug": a.aug,
         "weights": str(out / "weights" / ("last.pt" if a.full else "best.pt"))}  # fmt: skip
    write_metrics(run, m)
    (run / "notes.md").write_text(
        f"# Tip-end model ({a.name})\n\nYOLO11n-pose (COCO init), 1 class, 1 keypoint, data pose_{a.data} "
        "(configs/tip_data.yaml, split by time). Checkpoint choice: val. Task metrics: scripts/eval_tip_pose.py.\n"
        f"\nRegenerate: build_tip_dataset.py outlines / build --name {a.data}, then this script.\n\n{m}\n"
    )
    print("weights:", m["weights"], "| run:", run, m)


if __name__ == "__main__":
    main()
