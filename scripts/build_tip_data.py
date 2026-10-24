"""Plate-view frames + geometric tip per frame for every train/val clip with a cached grid.

Output: data/processed/tip_data/<clip>.npz
  views (T,168,224,3) uint8, has_grid (T,)
  tip (T,2) plate-view px of the white shaft end (nan if none), traced (T,2) clear tip end traced past it,
  depth (T,), cover (T,96) share of each well covered by foreground, fg_area (T,) foreground share of well area.
Regenerable; test plates refused.

    python scripts/build_tip_data.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_plate_grids import OUT as GRID_DIR  # noqa: E402
from cache_plate_grids import load_grid_windows  # noqa: E402

from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clips_for_split, load_splits, plate_of  # noqa: E402
from pipette_coach.localizer.plate_grid import window_at  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.tracking import plate_view as pv  # noqa: E402
from pipette_coach.tracking.tip_geom import (  # noqa: E402
    background,
    foreground,
    lab_diff,
    shaft_axis,
    tip_end,
    trace_tip,
)

OUT = PROCESSED_DIR / "tip_data"


def cell_masks(r_pitch: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : pv.H, 0 : pv.W]
    return np.stack([(xx - x) ** 2 + (yy - y) ** 2 <= (r_pitch * pv.PX) ** 2 for x, y in pv.NODES])


def main() -> None:
    cfg = load_config("tracking")["tip_geom"]
    splits = load_splits()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = cell_masks(cfg["cell_radius_pitch"])
    x0, y0 = pv.to_view(np.array([[-0.5, -0.5]]))[0].astype(int)
    x1, y1 = pv.to_view(np.array([[11.5, 7.5]]))[0].astype(int)
    t0 = time.time()
    clips = [c for s in ("train", "val") for c in clips_for_split(s) if plate_of(c) not in splits["test"]]
    clips = [c for c in clips if (GRID_DIR / f"{c}.json").exists()]
    # pass 1: plate views; per-clip median view
    med: dict[str, np.ndarray] = {}
    for clip in clips:
        wins = load_grid_windows(clip)
        frames = read_frames(clip)
        views = np.zeros((len(frames), pv.H, pv.W, 3), np.uint8)
        has = np.zeros(len(frames), bool)
        for i, f in enumerate(frames):
            w = window_at(wins, i)
            if w is not None and w["H_grid"] is not None:
                views[i], has[i] = pv.rectify(f, w["H_grid"]), True
        if has.any():
            med[clip] = background([views[i] for i in np.flatnonzero(has)])
        np.savez(OUT / f"{clip}.npz", views=views, has_grid=has)
    print(f"pass 1 done ({time.time() - t0:.0f}s)", flush=True)
    # background per plate: the plate view aligns the plate across clips, so the median over the plate's clips
    # removes the pipette (it sits somewhere else in each clip) - the per-clip median keeps it when the pipette
    # stays over the plate most of a clip
    bgs = {pl: background([m for c, m in med.items() if plate_of(c) == pl]) for pl in {plate_of(c) for c in med}}
    for clip in clips:
        d = np.load(OUT / f"{clip}.npz")
        views, has = d["views"], d["has_grid"]
        T = len(views)
        tip = np.full((T, 2), np.nan)
        traced = np.full((T, 2), np.nan)
        depth = np.zeros(T)
        cover = np.zeros((T, 96), np.float16)
        fg_area = np.zeros(T)
        if plate_of(clip) in bgs:
            bg = bgs[plate_of(clip)]
            for i in np.flatnonzero(has):
                m = foreground(views[i], bg, cfg["fg_thr"])
                cover[i] = (cells & m[None].astype(bool)).sum((1, 2)) / cells.sum((1, 2))
                fg_area[i] = m[y0:y1, x0:x1].mean()
                r = tip_end(m, cfg["min_depth_pitch"])
                if r is not None:
                    tip[i], depth[i] = r
                    u = shaft_axis(m, r[0])
                    traced[i] = r[0] if u is None else trace_tip(lab_diff(views[i], bg), r[0], u, cfg)
        np.savez_compressed(OUT / f"{clip}.npz", views=views, has_grid=has, tip=tip, traced=traced, depth=depth,
                            cover=cover, fg_area=fg_area)  # fmt: skip
        print(f"{clip}: {has.sum()}/{T} frames with grid, tip in {np.isfinite(tip[:, 0]).sum()} "
              f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    for pl, bg in bgs.items():
        cv2.imwrite(str(OUT / f"background_plate_{pl}.png"), bg)


if __name__ == "__main__":
    main()
