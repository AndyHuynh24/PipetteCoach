"""Which wells of a hand-labelled frame are visible, for training the well net (scripts/build_well_labels.py).

A well counts as visible (a positive) only if
  1. its rim or ring is seen near the labelled grid position (opening circle fit, or the ring filter), and
  2. it looks like itself: each labelled frame is warped flat with its own grid, and the well's patch is compared with
     the median of the same well in other labelled frames where the plate sat in the same place (within win_s seconds,
     grids within same_place wells). Change = 90th percentile of the per-pixel LAB difference (blur 1.5 px, best of
     +-2 px shifts so a slightly-off grid or the plate's parallax is not a change). Pipettes, gloves and thin tips
     passing over a well change it; a pipette held still for a few seconds is a minority of a 60 s window.
Anything else is IGNORED in training (neither well nor not-a-well). Frames with fewer than min_refs reference
frames (plate only briefly at that place, e.g. being put down) are not used at all. Wells being filled also read as
changed - they are ignored, which is the safe side.
"""

import copy

import cv2
import numpy as np

from .geometry import GRID, grid_error, pitch_px, plate_bbox, project
from .openings import opening_centres
from .rings import find_well_candidates

C = 24  # flat px per well spacing
_A = np.array([[C, 0, C], [0, C, C], [0, 0, 1]], float)  # canonical (col, row) -> flat px, 1-well margin
SHIFT = 2
THRESHOLD = 30.0


def flat_lab(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Plate warped flat (LAB float): well (c, r) centred at ((c + 1) C, (r + 1) C)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    F = cv2.warpPerspective(lab, _A @ np.linalg.inv(H), (13 * C, 9 * C)).astype(np.float32)
    return cv2.GaussianBlur(F, (0, 0), 1.5)


def well_cells(F: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """(96, 0.8 C, 0.8 C, 3) patches around the well centres of a flat plate, shifted by (dx, dy)."""
    r = int(0.4 * C)
    return np.stack([F[(rr + 1) * C - r + dy : (rr + 1) * C + r + dy, (cc + 1) * C - r + dx : (cc + 1) * C + r + dx]
                     for cc, rr in GRID.astype(int)])  # fmt: skip


def change(F: np.ndarray, refs: list[np.ndarray]) -> np.ndarray:
    """(96,) change of each well of flat plate F vs the per-well median of the reference flats."""
    ref = np.median(np.stack([well_cells(R) for R in refs]), 0)
    return np.min(
        [np.percentile(np.abs(well_cells(F, dx, dy) - ref).sum(3).reshape(len(GRID), -1), 90, axis=1)
         for dx in range(-SHIFT, SHIFT + 1) for dy in range(-SHIFT, SHIFT + 1)],
        axis=0,
    )  # fmt: skip


def reference_frames(items: list[tuple[int, np.ndarray, float]], k: int, win_s: float = 30.0,
                     same_place: float = 0.3) -> list[int]:  # fmt: skip
    """Indices into items (frame index, H, t) usable as the clean look for items[k]: within win_s seconds and with the
    plate in the same place (grid within same_place well spacings)."""
    _, H, t = items[k]
    return [j for j, (_, Hj, tj) in enumerate(items)
            if abs(tj - t) <= win_s and grid_error(Hj, H, ignore_labelling=False) <= same_place]  # fmt: skip


def rim_evidence(frame: np.ndarray, H: np.ndarray, cfg: dict) -> np.ndarray:
    """(96,) bool: the well's opening circle or ring is seen near the grid position."""
    g, p = project(H, GRID), pitch_px(H)
    rcfg = copy.deepcopy(cfg)
    rcfg["wells"]["method"] = "ring"
    P, _, _ = find_well_candidates(frame, rcfg, plate_bbox(H, 1.0))
    ring = np.linalg.norm(g[:, None] - P[None], axis=2).min(1) < 0.3 * p if len(P) else np.zeros(len(GRID), bool)
    return ring | np.isfinite(opening_centres(frame, H, cfg)[:, 0])
