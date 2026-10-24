"""YOLO dataset for pipette + tip detection on PLATE CROPS (two classes, as in the KicoSVK YOLOv7 project:
"the entire pipette and the tip of this pipette").

Crop: the plate (from the cached grid) + margin, at native resolution (the whole plate is ~200 x 140 px of a
1920 x 1080 frame; a crop lets YOLO see the tip at ~3x the pixels).
Labels (pseudo, no hand work): plate-view foreground vs the plate's clean background (tracking/tip_geom.py); the
entering object -> class 0 "pipette" box (its part inside the crop); traced tip end -> class 1 "tip" box (0.8 pitch).
Only frames where the object is long and thin (a pipette, not a hand) are kept; frames with no foreground in the
well area are empty negatives. Hand-clicked frames (data/annotations/tip_clicks.json, scripts/label_tips.py)
take precedence: the clicked tip(s) are the tip boxes, the foreground object(s) touching them give the pipette
box (no box drawing needed), and "no tip" clicks are negatives - including frames the pseudo-labels had to skip.
Train plates only; YOLO's own val = held-out train clips (real evaluation: val plate 4, never trained on).

    python scripts/build_pipette_yolo_dataset.py
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_plate_grids import load_grid_windows  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, plate_of  # noqa: E402
from pipette_coach.localizer.geometry import pitch_px  # noqa: E402
from pipette_coach.localizer.plate_grid import window_at  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.tracking import plate_view as pv  # noqa: E402
from pipette_coach.tracking.tip_data import load_clip, split_clips  # noqa: E402
from pipette_coach.tracking.tip_geom import entering_object, foreground  # noqa: E402

OUT = PROCESSED_DIR / "pipette_yolo"
# canonical crop corners (pitch) = exactly the plate-view region, so a pipette box that is cut at the crop edge is
# cut where its foreground mask ends too (the mask only exists inside the plate view)
CROP = np.array([[pv.X0, pv.Y0], [11 - pv.X0, pv.Y0], [11 - pv.X0, 7 - pv.Y0], [pv.X0, 7 - pv.Y0]])
TIP_BOX = 0.8  # pitch
CLICKS = REPO_ROOT / "data" / "annotations" / "tip_clicks.json"


def crop_rect(H: np.ndarray, shape) -> tuple[int, int, int, int]:
    c = cv2.perspectiveTransform(CROP[None], H)[0]
    x0, y0 = np.floor(c.min(0)).astype(int)
    x1, y1 = np.ceil(c.max(0)).astype(int)
    return max(x0, 0), max(y0, 0), min(x1, shape[1]), min(y1, shape[0])


def elongation(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return 0.0
    P = np.stack([xs, ys], 1).astype(np.float64)
    ev = np.linalg.eigvalsh(np.cov((P - P.mean(0)).T))
    return float(np.sqrt(ev[1] / max(ev[0], 1e-6)))


def near_tips(mask: np.ndarray, tips_view: np.ndarray, radius_px: float) -> np.ndarray:
    """Union of the foreground components that come within radius_px of any clicked tip (plate-view px)."""
    n, lab = cv2.connectedComponents(mask)
    keep = np.zeros(mask.shape, bool)
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    for tx, ty in tips_view:
        near = (xx - tx) ** 2 + (yy - ty) ** 2 <= radius_px**2
        for k in np.unique(lab[near & (lab > 0)]):
            keep |= lab == k
    return keep


def yolo_line(cls: int, box: tuple, w: int, h: int) -> str:
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    return f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def main() -> None:
    cfg = load_config("tracking")
    tg = cfg["tip_geom"]
    clicks = json.loads(CLICKS.read_text())["labels"] if CLICKS.exists() else {}
    if OUT.exists():
        shutil.rmtree(OUT)
    clips = split_clips("train")
    rng = np.random.default_rng(0)
    held = set(rng.choice(clips, max(1, len(clips) // 7), replace=False).tolist())  # YOLO's own val (monitoring)
    bgs = {}
    stats = {"pipette": 0, "empty": 0, "skipped": 0, "clicked_tips": 0, "clicked_none": 0}
    for clip in clips:
        split = "val" if clip in held else "train"
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        pl = plate_of(clip)
        if pl not in bgs:
            bgs[pl] = cv2.imread(str(PROCESSED_DIR / "tip_data" / f"background_plate_{pl}.png"))
        d = load_clip(clip)
        frames = read_frames(clip)
        wins = load_grid_windows(clip)
        clicked = {int(f) for f in clicks.get(clip, {})}
        every2 = set(np.flatnonzero(d["has_grid"])[::2].tolist())  # neighbours are near-duplicates
        for i in sorted(every2 | {f for f in clicked if d["has_grid"][f]}):  # ... but every clicked frame counts
            w = window_at(wins, i)
            H = w["H_grid"]
            Minv = np.linalg.inv(pv.view_matrix(H))
            x0, y0, x1, y1 = crop_rect(H, frames[i].shape)
            crop = frames[i][y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            m = foreground(d["views"][i], bgs[pl], tg["fg_thr"])
            obj = entering_object(m, tg["min_depth_pitch"])
            lines = []
            click = clicks.get(clip, {}).get(str(i))
            half = TIP_BOX / 2 * pitch_px(H)
            pmask, tips_img = None, None
            if click and click["kind"] == "none":  # you said: no tip over the plate -> negative
                stats["clicked_none"] += 1
            elif click:  # your tip click picks the pipette: the foreground object(s) touching the clicked tip(s)
                tips_img = np.array(click["tips_img"], float)
                pmask = near_tips(m, np.array(click["tips_view"], float), 1.5 * pv.PX)
                stats["clicked_tips"] += 1
            elif obj is None and d["fg_area"][i] < cfg["labels_b"]["max_empty_fg"]:
                stats["empty"] += 1
            elif obj is not None and elongation(obj[0]) >= 3.0:  # pseudo-label: a long thin entering object
                pmask = obj[0]
                t = d["traced"][i] if np.isfinite(d["traced"][i, 0]) else obj[1]
                tips_img = cv2.perspectiveTransform(np.array([[t]], np.float64), Minv)[0]
                stats["pipette"] += 1
            else:
                stats["skipped"] += 1
                continue
            if tips_img is not None:
                pbox = [np.inf, np.inf, -np.inf, -np.inf]
                if pmask is not None and pmask.any():
                    comp = cv2.warpPerspective(pmask.astype(np.uint8), Minv, (frames[i].shape[1], frames[i].shape[0]),
                                               flags=cv2.INTER_NEAREST)  # fmt: skip
                    ys, xs = np.nonzero(comp[y0:y1, x0:x1])
                    if len(xs):
                        pbox = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
                for tx, ty in tips_img:
                    tb = (tx - x0 - half, ty - y0 - half, tx - x0 + half, ty - y0 + half)
                    lines.append(yolo_line(1, tb, cw, ch))
                    pbox = [min(pbox[0], tb[0]), min(pbox[1], tb[1]), max(pbox[2], tb[2]), max(pbox[3], tb[3])]
                lines.insert(0, yolo_line(0, tuple(pbox), cw, ch))  # the pipette box includes its (clear) tip
            name = f"{clip}_f{i:03d}"
            cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: pipette\n  1: tip\n"
    )
    print(f"{stats}; held-out clips for YOLO val: {sorted(held)}")


if __name__ == "__main__":
    main()
