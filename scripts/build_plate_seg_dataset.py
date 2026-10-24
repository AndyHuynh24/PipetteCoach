"""YOLO segmentation dataset for the plate (1 class), labels from the cached measured plate outlines.

Outline per frame = the measured plate box of its static window (data/processed/plate_grids/, the approved
edges + wells method; amodal: the full plate even where a hand covers it). Frames without a known outline are
left out: they still show the plate (e.g. at the frame edge), and an empty label would teach "not a plate".
A share of frames is also added ROTATED 90 / 270 deg (image and outline), so upright plates are well represented
(our cameras only saw plates lying sideways); training adds any-angle, scale and perspective augmentation on top.
Train plates -> train; val plate 4 -> YOLO val (monitoring). Optional: hand-labelled / auto-labelled video frames.

    python scripts/build_plate_seg_dataset.py
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_plate_grids import OUT as GRID_DIR  # noqa: E402
from cache_plate_grids import load_grid_windows  # noqa: E402

from pipette_coach.data import PROCESSED_DIR, clips_for_split, load_splits, plate_of  # noqa: E402
from pipette_coach.localizer.plate_grid import window_at  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402

OUT = PROCESSED_DIR / "plate_seg"
EVERY = 4  # frames per clip: every 4th (neighbours are near-duplicates)
ROT_SHARE = 0.4  # share of frames also added rotated (half 90, half 270 deg)


def rot90(img: np.ndarray, poly: np.ndarray | None, k: int):
    """Rotate image by k*90 deg counter-clockwise and the polygon (px) with it."""
    h, w = img.shape[:2]
    out = np.rot90(img, k).copy()
    if poly is None:
        return out, None
    p = poly.copy()
    for _ in range(k % 4):
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p = np.stack([y, w - 1 - x], 1)  # (x, y) -> (y, W-1-x) for one CCW turn
        h, w = w, h
    return out, p


def line(poly: np.ndarray, w: int, h: int) -> str:
    q = np.clip(poly / [w, h], 0, 1)
    return "0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in q)


def add_video(stats: dict, labels_path: Path, train_until_s: float, every: int, rng) -> None:
    """Frames of an auto-labelled video (scripts/autolabel_video_plate.py): chunks before train_until_s -> train
    (+ rotated copies), later chunks -> val (held out in time; same camera, so a check, not an unseen camera)."""
    import json

    lab = json.loads(labels_path.read_text())
    cap = cv2.VideoCapture(lab["video"])
    fps = lab["fps"]
    for ch in lab["chunks"]:
        split = "train" if ch["start"] / fps < train_until_s else "val"
        poly = np.array(ch["outline"], float)
        for i in range(ch["start"], ch["end"], every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                continue
            variants = [(0, f, poly)]
            if split == "train" and rng.random() < ROT_SHARE:
                k = int(rng.choice([1, 3]))
                variants.append((k, *rot90(f, poly, k)))
            for k, img, p in variants:
                name = f"{labels_path.stem}_f{i:05d}_r{k * 90}"
                h, wd = img.shape[:2]
                cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (OUT / "labels" / split / f"{name}.txt").write_text(line(p, wd, h) + "\n")
                stats[split][0] += 1
                stats[split][2] += k != 0


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video-labels", default="", help="comma-separated data/processed/video_labels/*.json")
    ap.add_argument("--video-train-until", type=float, default=300.0, help="seconds; later chunks go to val")
    ap.add_argument("--video-every", type=int, default=10, help="every n-th frame of a labelled chunk")
    a = ap.parse_args()
    splits = load_splits()
    if OUT.exists():
        shutil.rmtree(OUT)
    rng = np.random.default_rng(0)
    stats = {"train": [0, 0, 0], "val": [0, 0, 0]}  # plate frames, empty frames, rotated copies
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for clip in clips_for_split(split):
            if plate_of(clip) in splits["test"] or not (GRID_DIR / f"{clip}.json").exists():
                continue
            wins = load_grid_windows(clip)
            frames = read_frames(clip)
            for i in range(0, len(frames), EVERY):
                w = window_at(wins, i) if wins else None
                if w is None or w["H_grid"] is None:
                    # no known outline does NOT mean no plate: these frames (plate 10, other camera pose) show the
                    # plate partly at the frame edge or held up - labelling them empty would teach "not a plate"
                    continue
                poly = np.asarray(w["box"], float)
                f = frames[i]
                variants = [(0, f, poly)]
                if split == "train" and rng.random() < ROT_SHARE:
                    k = int(rng.choice([1, 3]))
                    variants.append((k, *rot90(f, poly, k)))
                for k, img, p in variants:
                    name = f"{clip}_f{i:03d}_r{k * 90}"
                    h, wd = img.shape[:2]
                    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    (OUT / "labels" / split / f"{name}.txt").write_text("" if p is None else line(p, wd, h) + "\n")
                    stats[split][0 if p is not None else 1] += 1
                    stats[split][2] += k != 0
    for vl in [x for x in a.video_labels.split(",") if x]:
        add_video(stats, Path(vl), a.video_train_until, a.video_every, rng)
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n")
    print({s: dict(zip(("plate", "empty", "rotated"), v, strict=True)) for s, v in stats.items()})


if __name__ == "__main__":
    main()
