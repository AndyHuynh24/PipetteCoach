"""Geometric pipette-tip finder in the plate view (used to bootstrap labels for the learned tip model).

Per static window the plate is still, so the median of its plate-view frames is the clean plate (a moving
pipette/hand drops out). Foreground = colour change vs that background. A pipette enters from outside the plate;
its tip end is the point of the entering object deepest inside the well area, measured from where it enters.
"""

import cv2
import numpy as np

from pipette_coach.tracking import plate_view as pv


def background(views: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(views), axis=0).astype(np.uint8)


def foreground(view: np.ndarray, bg: np.ndarray, thr: float) -> np.ndarray:
    a = cv2.cvtColor(cv2.GaussianBlur(view, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(bg, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    m = (np.linalg.norm(a - b, axis=2) > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def entering_object(mask: np.ndarray, min_depth_pitch: float = 0.5) -> tuple[np.ndarray, np.ndarray, float] | None:
    """The foreground object that enters from the view border and reaches deepest into the well area.
    Returns (its mask, its deepest point = tip end, depth into the well area in pitch) or None."""
    n, lab = cv2.connectedComponents(mask)
    border = np.zeros_like(mask, bool)
    b = int(pv.PX * 0.5)
    border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
    x0, y0 = pv.to_view(np.array([[-0.5, -0.5]]))[0]
    x1, y1 = pv.to_view(np.array([[11.5, 7.5]]))[0]
    best = None
    for k in range(1, n):
        comp = lab == k
        if not (comp & border).any():
            continue  # not entering from outside
        ys, xs = np.nonzero(comp)
        ey, ex = np.nonzero(comp & border)
        entry = np.array([ex.mean(), ey.mean()])
        inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        if not inside.any():
            continue
        P = np.stack([xs[inside], ys[inside]], 1).astype(np.float64)
        far = P[np.argmax(np.linalg.norm(P - entry, axis=1))]
        depth = min(far[0] - x0, x1 - far[0], far[1] - y0, y1 - far[1]) / pv.PX
        if depth >= min_depth_pitch and (best is None or depth > best[2]):
            best = (comp, far, float(depth))
    return best


def tip_end(mask: np.ndarray, min_depth_pitch: float = 0.5) -> tuple[np.ndarray, float] | None:
    """Deepest point inside the well area of a foreground object that enters from the view border.
    Returns (plate-view point, depth into the well area in pitch) or None."""
    r = entering_object(mask, min_depth_pitch)
    return None if r is None else (r[1], r[2])


def lab_diff(view: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(cv2.GaussianBlur(view, (0, 0), 0.8), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(bg, (0, 0), 0.8), cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.linalg.norm(a - b, axis=2)


def shaft_axis(mask: np.ndarray, end: np.ndarray, back_pitch: float = 2.0) -> np.ndarray | None:
    """Unit direction of the shaft at its end (pointing toward the end): PCA of the foreground within
    back_pitch of the end, oriented away from the bulk of the object."""
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(np.float64)
    near = P[np.linalg.norm(P - end, axis=1) < back_pitch * pv.PX]
    if len(near) < 10:
        return None
    c = near.mean(0)
    _, _, vt = np.linalg.svd(near - c, full_matrices=False)
    u = vt[0]
    return u if np.dot(end - c, u) >= 0 else -u


def trace_tip(diff: np.ndarray, end: np.ndarray, u: np.ndarray, cfg: dict) -> np.ndarray:
    """Follow the faint clear tip past the shaft end along the shaft axis u: the tip end is the last point whose
    difference from the clean plate (mean over a thin band across the axis) stays above the noise level; short
    gaps are bridged. Noise = robust level of the difference over the whole view."""
    noise = float(np.median(diff))
    mad = float(np.median(np.abs(diff - noise))) + 1e-6
    thr = noise + cfg["trace_k_mad"] * mad
    n = np.array([-u[1], u[0]])
    ts = np.arange(0.0, cfg["trace_max_pitch"] * pv.PX, 0.5)
    offs = np.linspace(-cfg["trace_band_px"], cfg["trace_band_px"], 5)
    P = end[None, None] + ts[:, None, None] * u[None, None] + offs[None, :, None] * n[None, None]
    prof = cv2.remap(diff, P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), cv2.INTER_LINEAR).max(1)
    prof = np.convolve(prof, np.ones(3) / 3, mode="same")
    last, gap = 0.0, 0.0
    for t, v in zip(ts, prof, strict=True):
        if v > thr:
            last, gap = t, 0.0
        else:
            gap += 0.5
            if gap > cfg["trace_gap_px"]:
                break
    return end + last * u
