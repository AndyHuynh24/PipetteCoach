"""Synthetic "lifted / tilted / any-size" plates for the plate segmentation model (exact labels, no hand work).

Source plates: dataset frames whose plate outline is known (data/processed/plate_grids/, the approved method).
Each plate is cut out along its outline (rectified to a flat image of the plate), then pasted into a background
frame with a random camera: any rotation, size 0.5-3.5x of the dataset plate, perspective tilt up to --max-tilt
degrees (a plate held up and tilted in a hand), random position (may be partly cut by the frame edge), brightness
/ blur change, and skin-coloured blobs over its edges (fingers holding it). The plate is clear plastic, so it is
blended slightly with the background. The label is the pasted outline (exact).
Backgrounds: dataset frames whose own plate outline is known; that plate is labelled too (frames without a known
outline may still show a plate, unlabelled, so they are not used). Train plates only.

    python scripts/build_plate_synth.py --n 1500
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_plate_seg_dataset import OUT as SEG_OUT  # noqa: E402
from build_plate_seg_dataset import line  # noqa: E402
from cache_plate_grids import OUT as GRID_DIR  # noqa: E402
from cache_plate_grids import load_grid_windows  # noqa: E402
from eval_generalization import camera_warp  # noqa: E402

from pipette_coach.data import clips_for_split  # noqa: E402
from pipette_coach.localizer.plate_grid import window_at  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402

PW, PH = 556, 382  # flat plate image (13.9 x 9.55 spacings at 40 px)
SKIN = [(160, 180, 230), (140, 165, 215), (120, 150, 200), (175, 195, 235)]  # BGR skin tones


def collect(rng, n_src: int):
    """Flat plate images (PW x PH) and background frames with their own plate outline (or None)."""
    plates, bgs = [], []
    for clip in clips_for_split("train"):
        if not (GRID_DIR / f"{clip}.json").exists():
            continue
        wins = load_grid_windows(clip)
        frames = read_frames(clip)
        for i in rng.choice(len(frames), min(3, len(frames)), replace=False):
            w = window_at(wins, int(i)) if wins else None
            box = None if w is None or w["H_grid"] is None else np.asarray(w["box"], np.float32)
            if box is not None:  # only backgrounds whose own plate is known (no unlabelled plate in the image)
                bgs.append((frames[i], box))
            if box is not None and len(plates) < n_src:
                M = cv2.getPerspectiveTransform(box, np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]]))
                plates.append(cv2.warpPerspective(frames[i], M, (PW, PH), flags=cv2.INTER_LINEAR))
    return plates, bgs


def paste(bg, plate, rng, max_tilt):
    h, w = bg.shape[:2]
    s = float(np.exp(rng.uniform(np.log(0.5), np.log(3.5))))  # plate long side ~ 200 px x s
    long_px = 200 * s
    flat = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])
    centre = np.array([PW / 2, PH / 2])
    dst = np.array([rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.1 * h, 0.9 * h)])
    Wp = camera_warp(centre, long_px / PW, rng.uniform(0, 360), rng.uniform(0, max_tilt), rng.uniform(0, 360), dst,
                     f=rng.uniform(900, 2500))  # fmt: skip
    quad = cv2.perspectiveTransform(flat[None], Wp)[0]
    warped = cv2.warpPerspective(plate, Wp, (w, h), flags=cv2.INTER_LINEAR)
    mask = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.8)[..., None]
    alpha = rng.uniform(0.8, 0.97)  # clear plastic: a little of the background shows through
    img = bg.astype(np.float32) * (1 - alpha * mask) + warped.astype(np.float32) * alpha * mask
    img = img * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(0, 4))):  # fingers over the plate's edge
        a = int(rng.integers(4))
        p = quad[a] + (quad[(a + 1) % 4] - quad[a]) * rng.uniform(0.1, 0.9)
        ax = (int(long_px * rng.uniform(0.05, 0.12)), int(long_px * rng.uniform(0.15, 0.35)))
        col = tuple(int(c + rng.integers(-15, 16)) for c in SKIN[int(rng.integers(len(SKIN)))])
        cv2.ellipse(img, (int(p[0]), int(p[1])), ax, rng.uniform(0, 180), 0, 360, col, -1, cv2.LINE_AA)
    return img, quad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--max-tilt", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    plates, bgs = collect(rng, 120)
    out_i, out_l = SEG_OUT / "images" / "train", SEG_OUT / "labels" / "train"
    out_i.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)
    for k in range(a.n):
        bg, own = bgs[int(rng.integers(len(bgs)))]
        img, quad = paste(bg, plates[int(rng.integers(len(plates)))], rng, a.max_tilt)
        h, w = img.shape[:2]
        lines = [line(quad, w, h)]
        if own is not None:
            m = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
            o = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(o, np.round(own).astype(np.int32), 1)
            if (m & o).sum() < 0.2 * o.sum():  # the background's own plate is still mostly visible: label it too
                lines.append(line(own, w, h))
            else:
                continue  # pasted over the real plate: ambiguous outline, skip
        cv2.imwrite(str(out_i / f"synth_{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_l / f"synth_{k:05d}.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(list(out_i.glob('synth_*.jpg')))} synthetic images in {out_i} ({len(plates)} source plates)")


if __name__ == "__main__":
    main()
