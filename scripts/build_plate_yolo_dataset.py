"""Build a YOLO oriented-box dataset from confident auto-labels.

train <- data/processed/plate_autolabels/<train-labels>.json, val <- .../val.json (val always uses the
grid-fit labels so v1 and v2 are scored on the same val set).
Output: data/processed/<out>/{images,labels}/{train,val}/ + data.yaml (regenerable, gitignored).

    python scripts/build_plate_yolo_dataset.py                                   # v1
    python scripts/build_plate_yolo_dataset.py --train-labels train_session --out plate_yolo_v2
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clip_path  # noqa: E402
from pipette_coach.localizer.geometry import OUTLINE, project  # noqa: E402

CFG = load_config("localizer")


def build(out: Path, split: str, label_name: str) -> int:
    labels = json.loads((PROCESSED_DIR / "plate_autolabels" / f"{label_name}.json").read_text())
    (out / "images" / split).mkdir(parents=True)
    (out / "labels" / split).mkdir(parents=True)
    n = 0
    for clip, frames in labels.items():
        good = {f["frame"]: f for f in frames if f["H"] is not None and f["frac"] >= CFG["grid_fit"]["confident_frac"]}
        if not good:
            continue
        cap = cv2.VideoCapture(str(clip_path(clip)))
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i in good:
                q = project(np.array(good[i]["H"]), OUTLINE) / [img.shape[1], img.shape[0]]
                name = f"{clip}_{i:03d}"
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / split / f"{name}.txt").write_text(
                    "0 " + " ".join(f"{v:.5f}" for v in q.ravel()) + "\n"
                )
                n += 1
            i += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-labels", default="train")
    ap.add_argument("--out", default="plate_yolo")
    a = ap.parse_args()
    out = PROCESSED_DIR / a.out
    if out.exists():
        shutil.rmtree(out)
    for split, name in (("train", a.train_labels), ("val", "val")):
        print(split, build(out, split, name), "images")
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n")


if __name__ == "__main__":
    main()
