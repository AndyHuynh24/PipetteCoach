"""Fixed top-down view of the plate from its grid H (canonical -> image).

Canonical x = column 0..11, y = row 0..7 (A1 = (0, 0)), in well spacings. The view covers the wells plus `margin`
spacings around them, at `px` pixels per spacing, so every well sits at the same pixel in every frame and every
clip (camera distance and pose drop out) and a pipette entering from outside is visible.
"""

import cv2
import numpy as np

from pipette_coach.localizer.geometry import GRID

PX = 14
MARGIN = 2.5
X0, Y0 = -MARGIN, -MARGIN
W = int(round((11 + 2 * MARGIN) * PX))  # 224
H = int(round((7 + 2 * MARGIN) * PX))  # 168


def view_matrix(H_grid: np.ndarray) -> np.ndarray:
    """Image -> plate view (3x3)."""
    S = np.array([[PX, 0, -X0 * PX], [0, PX, -Y0 * PX], [0, 0, 1]], np.float64)
    return S @ np.linalg.inv(np.asarray(H_grid, np.float64))


def rectify(bgr: np.ndarray, H_grid: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(
        bgr, view_matrix(H_grid), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def to_view(canon: np.ndarray) -> np.ndarray:
    """Canonical (x, y) -> plate-view pixels."""
    return (np.asarray(canon, np.float64) - [X0, Y0]) * PX


def to_canon(px: np.ndarray) -> np.ndarray:
    return np.asarray(px, np.float64) / PX + [X0, Y0]


NODES = to_view(GRID)  # (96, 2) well centres in the plate view; index = row * 12 + col


def nearest_well(px: np.ndarray, max_pitch: float = 0.5) -> int | None:
    """Index of the well whose centre is nearest to a plate-view point, if within max_pitch."""
    d = np.linalg.norm(NODES - np.asarray(px, np.float64), axis=1)
    k = int(np.argmin(d))
    return k if d[k] <= max_pitch * PX else None
