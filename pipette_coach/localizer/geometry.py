"""Plate geometry: the canonical 12x8 grid, projections, orientation, and grid comparison."""

import cv2
import numpy as np

N_COLS, N_ROWS = 12, 8
# Canonical well centres (x = column index, y = row index), row-major A1..H12. Same convention as
# protocol/wells.py; only used internally as an array.
GRID = np.array([(c, r) for r in range(N_ROWS) for c in range(N_COLS)], np.float32)
CENTER = np.array([(N_COLS - 1) / 2, (N_ROWS - 1) / 2], np.float32)
# SBS plate outline in pitch units: A1 centre is 14.38 mm (1.6 pitch) from the short edge and
# 11.24 mm (1.25 pitch) from the long edge; symmetric on the opposite sides.
OUTLINE = np.array([[-1.6, -1.25], [12.6, -1.25], [12.6, 8.25], [-1.6, 8.25]], np.float32)

# Relabelling transforms in canonical space.
ROT180 = np.array([[-1, 0, N_COLS - 1], [0, -1, N_ROWS - 1], [0, 0, 1]], np.float64)
FLIP_ROWS = np.array([[1, 0, 0], [0, -1, N_ROWS - 1], [0, 0, 1]], np.float64)


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map canonical points (N, 2) to image pixels with H."""
    return cv2.perspectiveTransform(np.asarray(pts, np.float32)[None], np.asarray(H, np.float64))[0]


def pitch_px(H: np.ndarray) -> float:
    """Well pitch in pixels near the plate centre."""
    p = project(H, np.stack([CENTER, CENTER + [1, 0], CENTER + [0, 1]]))
    return float((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 2)


def axis_pitches(H: np.ndarray) -> tuple[float, float]:
    """(column pitch, row pitch) in pixels near the plate centre."""
    p = project(H, np.stack([CENTER, CENTER + [1, 0], CENTER + [0, 1]]))
    return float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[2] - p[0]))


def is_plausible(H: np.ndarray, pitch_range: tuple[float, float], max_axis_ratio: float) -> bool:
    """Grid has a believable size and is not squashed (wells are on a square lattice)."""
    c, r = axis_pitches(H)
    ratio = max(c, r) / max(min(c, r), 1e-6)
    return pitch_range[0] <= (c + r) / 2 <= pitch_range[1] and ratio <= max_axis_ratio


def is_mirrored(H: np.ndarray) -> bool:
    """True if H maps the canonical grid with flipped handedness (impossible for a camera looking down)."""
    p = project(H, np.stack([CENTER, CENTER + [1, 0], CENTER + [0, 1]]))
    a, b = p[1] - p[0], p[2] - p[0]
    return float(a[0] * b[1] - a[1] * b[0]) < 0


ROW_DIRS = {"down": (0.0, 1.0), "up": (0.0, -1.0), "right": (1.0, 0.0), "left": (-1.0, 0.0)}


def orient(
    H: np.ndarray,
    a1_rule: str = "image_top_left",
    camera_mirrored: bool = False,
    prev_H: np.ndarray | None = None,
    rows_toward: str | None = None,
) -> np.ndarray:
    """Fix the labelling ambiguity of a fitted grid.

    1. Handedness: a camera looking down sees the plate un-mirrored (unless the camera image itself is
       mirrored), so a mirrored fit is relabelled by flipping rows.
    2. 180-degree ambiguity, in order of preference: `rows_toward` (known image direction of rows A -> H,
       e.g. from calibration or configs/plate_orientation.json), else the labelling closest to a previous
       H, else a1_rule (a guess: only right for plates lying the usual way, A1 top-left).
    """
    H = np.asarray(H, np.float64)
    if is_mirrored(H) != camera_mirrored:
        H = H @ FLIP_ROWS
    options = [H, H @ ROT180]
    if rows_toward is not None:
        d = np.array(ROW_DIRS[rows_toward])

        def row_alignment(h: np.ndarray) -> float:
            a, b = project(h, np.array([[5.5, 0.0], [5.5, 7.0]]))
            return float(np.dot(b - a, d))

        return max(options, key=row_alignment)
    if prev_H is not None:
        return min(
            options, key=lambda h: float(np.linalg.norm(project(h, GRID) - project(prev_H, GRID), axis=1).mean())
        )
    if a1_rule == "image_top_left":
        return min(options, key=lambda h: float(project(h, GRID[:1])[0].sum()))
    raise ValueError(f"unknown a1_rule: {a1_rule}")


def grid_error(H_a: np.ndarray, H_b: np.ndarray, ignore_labelling: bool = True) -> float:
    """Mean distance between the 96 projected well centres of two fits, in pitch units of H_a.

    With ignore_labelling, compares geometry only (min over 180-degree flip and mirror relabellings),
    which is what matters for "is the grid on the wells"; orientation is checked separately.
    """
    a = project(H_a, GRID)
    b = project(H_b, GRID).reshape(N_ROWS, N_COLS, 2)
    syms = [b, b[::-1, ::-1], b[::-1], b[:, ::-1]] if ignore_labelling else [b]
    best = min(float(np.linalg.norm(a - s.reshape(-1, 2), axis=1).mean()) for s in syms)
    return best / pitch_px(H_a)


def quad_to_H(quad: np.ndarray) -> np.ndarray:
    """Homography from an oriented plate box (4 image corners, any start/winding) to canonical grid.
    The long side is taken as the 12-column direction. Orientation is fixed later by orient()."""
    q = np.asarray(quad, np.float32).reshape(4, 2)
    if np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[2] - q[1]):
        q = np.roll(q, -1, axis=0)
    H = cv2.getPerspectiveTransform(OUTLINE, q).astype(np.float64)
    # corner order can wind either way; a downward camera never sees the plate mirrored, and the snap's
    # similarity fit cannot represent a mirror, so un-mirror here
    return H @ FLIP_ROWS if is_mirrored(H) else H


def plate_bbox(H: np.ndarray, margin_pitch: float = 0.0) -> tuple[int, int, int, int]:
    """Axis-aligned image bbox (x0, y0, x1, y1) of the plate outline grown by margin_pitch."""
    m = margin_pitch
    out = OUTLINE + np.array([[-m, -m], [m, -m], [m, m], [-m, m]], np.float32)
    p = project(H, out)
    x0, y0 = np.floor(p.min(0)).astype(int)
    x1, y1 = np.ceil(p.max(0)).astype(int)
    return int(x0), int(y0), int(x1), int(y1)
