"""Recolour the liquid INSIDE the pipette tip into other liquid colours (tip-model colour augmentation preview).

Liquid = pixels of the video's liquid hue range (configs/tip_data.yaml `liquid: [hue_lo, hue_hi, sat_min]`, OpenCV
hue 0-180, lo > hi wraps through red) within ROI px of the hand-labelled tip end, keeping only the coloured blobs that
touch the tip end - so hands, wells and the cup are left alone. Each liquid pixel's hue is moved to the target colour,
keeping its saturation / brightness pattern (meniscus, reflections); "clear" keeps only a trace of colour.
Output: a video of the labelled frames only (what training sees), tip crops of the original + every colour.

    python scripts/render_liquid_colors.py --video pipette6
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.tips import (  # noqa: E402
    LIQUID_COLOURS,
    liquid_centre,
    parse_cvat_points,
    recolour,
    tip_liquid_mask,
    video_path,
)


def crop(img: np.ndarray, tip: tuple[float, float], w: int, h: int, text: str) -> np.ndarray:
    """w x h window centred a bit behind the tip end (the liquid column), labelled."""
    x0, y0 = int(tip[0]) - w * 2 // 3, int(tip[1]) - h // 2
    p = cv2.copyMakeBorder(img, h, h, w, w, cv2.BORDER_CONSTANT)[y0 + h : y0 + 2 * h, x0 + w : x0 + 2 * w]
    p = cv2.resize(p, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(p, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, th, cv2.LINE_AA)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="own video name in configs/tip_data.yaml")
    ap.add_argument("--fps", type=float, default=15.0, help="preview speed (labelled frames per second)")
    ap.add_argument("--out", default=None, help="default runs/liquid_colors_<video>.mp4")
    a = ap.parse_args()
    v = load_config("tip_data")["videos"][a.video]
    tips, _ = parse_cvat_points(REPO_ROOT / v["xml"])
    centre = liquid_centre(v["liquid"])
    cap = cv2.VideoCapture(str(video_path(a.video)))
    if not cap.isOpened():
        sys.exit(f"can't open {video_path(a.video)}")
    out = Path(a.out) if a.out else REPO_ROOT / "runs" / f"liquid_colors_{a.video}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, empty = None, 0
    for f in sorted(tips):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        tip = tips[f][0]  # ponytail: first tip only; pipette7 has a few 2-tip frames
        m = tip_liquid_mask(img, tip, v["liquid"])
        empty += m.max() < 0.5
        tiles = [crop(img, tip, 200, 150, f"original f{f}")]
        tiles += [crop(recolour(img, m, centre, hh, g), tip, 200, 150, n) for n, (hh, g) in LIQUID_COLOURS.items()]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
        grid = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), a.fps, (grid.shape[1], grid.shape[0]))
        writer.write(grid)
    if writer is not None:
        writer.release()
    print(f"-> {out}  ({len(tips)} labelled frames, {empty} with no liquid found at the tip)")


if __name__ == "__main__":
    main()
