"""YOLO DETECTION dataset for the plate (1 class, axis-aligned boxes): "which object is the plate", any size.

Label meaning: the VISIBLE plate (hand-labelled frames: the visible part when an arm covers it). The full plate
is rebuilt later by geometry (wells + the plate's measured 13.9 x 9.5 spacing ratio), so YOLO only has to find it.

Sources:
  - hand-labelled frames, Ultralytics Platform NDJSON export (e.g. data/own/elastic-mammoth.ndjson + images
    downloaded next to it): frames with time < --train-until s -> train, later -> TEST (written to a separate
    folder, never used for training or checkpoint choice). Tiny stray boxes inside another box are dropped.
  - dataset train plates: box of the measured plate outline (data/processed/plate_grids/), clipped to the image;
    + upright (90 / 270 deg) copies of a share of them.
  - synthetic tilted / lifted plates (data/processed/plate_seg/images/train/synth_*): box of the pasted outline.
  - own videos with CVAT "Plate" boxes / masks (configs/tip_data.yaml, --cvat-videos, e.g. pipette5), split by
    their time split: train (+ upright copies), val (checkpoint choice), test -> plate_det_test_<video>/.
  - YOLO val (checkpoint choice): val plate 4 frames (dataset) + the own videos' val frames, never test frames.

    python scripts/build_plate_det_dataset.py --ndjson data/own/elastic-mammoth.ndjson --video-fps-step 6
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_plate_seg_dataset import rot90  # noqa: E402
from cache_plate_grids import OUT as GRID_DIR  # noqa: E402
from cache_plate_grids import load_grid_windows  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clips_for_split, load_splits, plate_of  # noqa: E402
from pipette_coach.localizer.plate_grid import window_at  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.tips import parse_cvat_plates, split_of  # noqa: E402

OUT = PROCESSED_DIR / "plate_det"
TEST = PROCESSED_DIR / "plate_det_test"
EVERY = 4
ROT_SHARE = 0.4


def box_line(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> str | None:
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def poly_box(poly: np.ndarray, w: int, h: int) -> str | None:
    return box_line(*poly.min(0), *poly.max(0), w, h)


def drop_stray(boxes: list[list[float]]) -> list[list[float]]:
    """Drop boxes that lie inside another, much bigger box (a stray click while annotating)."""
    keep = []
    for i, b in enumerate(boxes):
        inside = any(j != i and o[3] * o[4] > 10 * b[3] * b[4] and abs(b[1] - o[1]) < o[3] / 2
                     and abs(b[2] - o[2]) < o[4] / 2 for j, o in enumerate(boxes))  # fmt: skip
        if not inside:
            keep.append(b)
    return keep


def write(split_dir: Path, split: str, name: str, img: np.ndarray, lines: list[str]) -> None:
    (split_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (split_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(split_dir / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (split_dir / "labels" / split / f"{name}.txt").write_text("".join(f"{x}\n" for x in lines))


def add_cvat_videos(names: list[str], stats: dict) -> None:
    """Own videos' hand-labelled plate frames: train gets every frame + both upright copies (one bench, few frames),
    val goes to the YOLO val, test to its own folder (never used for training or checkpoint choice)."""
    vids = load_config("tip_data")["videos"]
    for name in names:
        v = vids[name]
        test = PROCESSED_DIR / f"plate_det_test_{name}"
        if test.exists():
            shutil.rmtree(test)
        cap = cv2.VideoCapture(str(REPO_ROOT / v["video"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for f, (x0, y0, x1, y1) in sorted(
            parse_cvat_plates(REPO_ROOT / v["xml"], tuple(v.get("plate_drop", []))).items()
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            sp, stem = split_of(f / fps, v), f"{name}_f{f:05d}"
            ln = [box_line(x0, y0, x1, y1, w, h)]
            if sp == "test":
                write(test, "test", stem, img, ln)
                continue
            write(OUT, sp, stem, img, ln)
            stats[f"{name}_{sp}"] = stats.get(f"{name}_{sp}", 0) + 1
            if sp == "train":
                corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                for kk in (1, 3):
                    r_img, r_poly = rot90(img, corners, kk)
                    write(OUT, "train", f"{stem}_r{kk * 90}", r_img, [poly_box(r_poly, r_img.shape[1], r_img.shape[0])])
                    stats["rotated"] += 1
        if test.exists():
            (test / "data.yaml").write_text(f"path: {test}\ntrain: images/test\nval: images/test\nnames:\n  0: plate\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", default="data/own/elastic-mammoth.ndjson")
    ap.add_argument(
        "--video-fps-step", type=float, default=6.0, help="seconds between exported frames (frame_001 = 0 s)"
    )
    ap.add_argument("--train-until", type=float, default=300.0, help="hand-labelled frames after this (s) = TEST")
    ap.add_argument("--cvat-videos", default="pipette5", help="own videos with CVAT plate labels (tip_data.yaml)")
    ap.add_argument("--cvat-only", action="store_true", help="only the --cvat-videos (skip ndjson, dataset, synth)")
    a = ap.parse_args()
    for d in (OUT, TEST):
        if d.exists():
            shutil.rmtree(d)
    rng = np.random.default_rng(0)
    stats = {"hand_train": 0, "hand_test": 0, "dataset": 0, "rotated": 0, "synth": 0, "val": 0, "stray_dropped": 0}
    add_cvat_videos([v for v in a.cvat_videos.split(",") if v], stats)
    if a.cvat_only:  # fine-tune on the own videos alone (other sources not on this machine)
        (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n")
        print(stats)
        return
    # hand-labelled frames
    nd = Path(a.ndjson)
    for x in [json.loads(line) for line in open(nd)][1:]:
        if x.get("type") != "image":
            continue
        k = int(re.search(r"frame_(\d+)", x["file"]).group(1))
        t = (k - 1) * a.video_fps_step
        img = cv2.imread(str(nd.parent / nd.stem / "images" / x["file"].replace(".webp", ".jpg")))
        h, w = img.shape[:2]
        boxes = x.get("annotations", {}).get("boxes", [])
        kept = drop_stray(boxes)
        stats["stray_dropped"] += len(boxes) - len(kept)
        lines = [box_line((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, w, h)
                 for _, cx, cy, bw, bh in kept]  # fmt: skip
        lines = [ln for ln in lines if ln]
        if t < a.train_until:
            write(OUT, "train", f"hand_{k:03d}_t{t:05.0f}", img, lines)
            stats["hand_train"] += 1
            for kk in (1, 3):  # upright / sideways copies of the hand-labelled frames too
                if rng.random() < 0.5:
                    corners = []
                    for _, cx, cy, bw, bh in kept:
                        x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
                        corners.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
                    r_img = np.rot90(img, kk).copy()
                    r_lines = [poly_box(rot90(img, c, kk)[1], r_img.shape[1], r_img.shape[0]) for c in corners]
                    write(OUT, "train", f"hand_{k:03d}_r{kk * 90}", r_img, [ln for ln in r_lines if ln])
                    stats["rotated"] += 1
        else:
            write(TEST, "test", f"hand_{k:03d}_t{t:05.0f}", img, lines)
            stats["hand_test"] += 1
    # dataset plates (train) and YOLO val (val plate 4)
    splits = load_splits()
    for split in ("train", "val"):
        for clip in clips_for_split(split):
            if plate_of(clip) in splits["test"] or not (GRID_DIR / f"{clip}.json").exists():
                continue
            wins = load_grid_windows(clip)
            frames = read_frames(clip)
            for i in range(0, len(frames), EVERY if split == "train" else 8):
                w_ = window_at(wins, i) if wins else None
                if w_ is None or w_["H_grid"] is None:
                    continue  # unknown outline: not labelled empty (the plate may still be in view)
                poly = np.asarray(w_["box"], float)
                f = frames[i]
                ln = poly_box(poly, f.shape[1], f.shape[0])
                if ln is None:
                    continue
                write(OUT, split, f"{clip}_f{i:03d}", f, [ln])
                stats["dataset" if split == "train" else "val"] += 1
                if split == "train" and rng.random() < ROT_SHARE:
                    kk = int(rng.choice([1, 3]))
                    r_img, r_poly = rot90(f, poly, kk)
                    write(OUT, "train", f"{clip}_f{i:03d}_r{kk * 90}", r_img,
                          [x for x in [poly_box(r_poly, r_img.shape[1], r_img.shape[0])] if x])  # fmt: skip
                    stats["rotated"] += 1
    # synthetic tilted / lifted plates (outlines -> boxes)
    syn = PROCESSED_DIR / "plate_seg"
    for img_p in sorted((syn / "images" / "train").glob("synth_*.jpg")):
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        lines = []
        for line in (syn / "labels" / "train" / f"{img_p.stem}.txt").read_text().splitlines():
            v = np.array(line.split()[1:], float).reshape(-1, 2) * [w, h]
            ln = poly_box(v, w, h)
            if ln:
                lines.append(ln)
        write(OUT, "train", img_p.stem, img, lines)
        stats["synth"] += 1
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n")
    (TEST / "data.yaml").write_text(f"path: {TEST}\ntrain: images/test\nval: images/test\nnames:\n  0: plate\n")
    print(stats)


if __name__ == "__main__":
    main()
