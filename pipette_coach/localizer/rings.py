"""Well candidates: wells look like a bright rim around a darker centre, so a ring-shaped matched
filter (positive annulus, negative inner disk) peaks at well centres."""

import cv2
import numpy as np


def ring_kernel(r_in: float, r_out: float) -> np.ndarray:
    """Zero-mean ring filter: +1/n on the annulus (r_in, r_out], -1/n inside r_in."""
    R = int(np.ceil(r_out)) + 1
    y, x = np.mgrid[-R : R + 1, -R : R + 1]
    d = np.hypot(x, y)
    k = np.zeros_like(d, np.float32)
    inner = d <= r_in
    ring = (d > r_in + 0.5) & (d <= r_out)
    k[inner] = -1.0 / inner.sum()
    k[ring] = 1.0 / ring.sum()
    return k


def ring_response(bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Per-pixel ring contrast (grey levels)."""
    w = cfg["wells"]
    p = w["pitch_px_nominal"]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.filter2D(g, -1, ring_kernel(w["ring_inner_frac"] * p, w["ring_outer_frac"] * p))


def _radial_kernels(r: float, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ring weights and the radial-direction products (ux^2, ux*uy, uy^2) on that ring."""
    R = int(np.ceil(r + 3 * sigma)) + 1
    y, x = np.mgrid[-R : R + 1, -R : R + 1].astype(np.float32)
    d = np.hypot(x, y)
    ring = np.exp(-0.5 * ((d - r) / sigma) ** 2)
    ring[d < 1e-6] = 0
    ring /= ring.sum()
    ux, uy = np.divide(x, d, out=np.zeros_like(x), where=d > 0), np.divide(y, d, out=np.zeros_like(y), where=d > 0)
    return ring, ring * ux * ux, ring * ux * uy, ring * uy * uy


def radial_edge_response(bgr: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(ratio, strength) maps. ratio = share of the colour-edge energy on a circle of the opening radius
    that points radially at the pixel (Di Zenzo colour structure tensor, so the sign of the edge and the
    colour of the fill do not matter); strength = that radial energy."""
    w = cfg["wells"]
    rc = w["radial_edge"]
    p = w["pitch_px_nominal"]
    im = cv2.GaussianBlur(bgr, (0, 0), 0.7).astype(np.float32)
    jxx = jxy = jyy = 0.0
    for k in range(3):
        gx = cv2.Sobel(im[..., k], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(im[..., k], cv2.CV_32F, 0, 1, ksize=3)
        jxx, jxy, jyy = jxx + gx * gx, jxy + gx * gy, jyy + gy * gy
    tot = jxx + jyy
    floor = float(np.percentile(tot, rc["noise_floor_pct"])) + 1e-6
    best_ratio = np.zeros(tot.shape, np.float32)
    best_strength = np.zeros(tot.shape, np.float32)
    for rf in rc["radii_frac"]:
        ring, kxx, kxy, kyy = _radial_kernels(rf * p, rc["ring_sigma_frac"] * p)
        rad = cv2.filter2D(jxx, -1, kxx) + 2 * cv2.filter2D(jxy, -1, kxy) + cv2.filter2D(jyy, -1, kyy)
        energy = cv2.filter2D(tot, -1, ring)
        ratio = rad / (energy + floor)  # noise floor keeps flat areas from scoring
        better = ratio > best_ratio
        best_ratio = np.where(better, ratio, best_ratio)
        best_strength = np.where(better, rad, best_strength)
    return best_ratio, best_strength


def find_well_candidates(
    bgr: np.ndarray, cfg: dict, roi: tuple[int, int, int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ring peaks. Returns (points (N, 2) in full-image pixels, scores (N,), response map of the ROI).

    roi = (x0, y0, x1, y1) restricts the search (clipped to the image); points are still full-image coords.
    """
    w = cfg["wells"]
    h, wd = bgr.shape[:2]
    x0, y0, x1, y1 = (0, 0, wd, h) if roi is None else roi
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, wd), min(y1, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32), np.zeros((0, 0), np.float32)
    img = bgr[y0:y1, x0:x1]
    nms = max(3, int(round(w["nms_frac"] * w["pitch_px_nominal"])))
    if w.get("method", "ring") == "radial_edge":
        ratio, _ = radial_edge_response(img, cfg)
        resp = ratio * 100.0  # score in "percent radial"; ring-based thresholds elsewhere use their own scale
        peak = (ratio == cv2.dilate(ratio, np.ones((nms, nms), np.uint8))) & (ratio > w["radial_edge"]["min_ratio"])
        use_sat = w["radial_edge"]["use_saturation_filter"]
    else:
        resp = ring_response(img, cfg)
        peak = (resp == cv2.dilate(resp, np.ones((nms, nms), np.uint8))) & (resp > w["min_ring_score"])
        use_sat = True
    if use_sat:
        sat = cv2.blur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1], (w["saturation_blur_px"],) * 2)
        peak &= sat < w["max_saturation"]
    ys, xs = np.nonzero(peak)
    pts = np.stack([xs + x0, ys + y0], 1).astype(np.float32)
    return pts, resp[ys, xs], resp


def sample(img: np.ndarray, pts: np.ndarray, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Nearest-pixel sample of img at full-image points (out-of-bounds clipped)."""
    x = np.clip(np.round(pts[:, 0] - offset[0]).astype(int), 0, img.shape[1] - 1)
    y = np.clip(np.round(pts[:, 1] - offset[1]).astype(int), 0, img.shape[0] - 1)
    return img[y, x]


def empty_map(bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """True where the image is visibly empty (bare bench / plate rim / paper): no texture, no colour, not dark."""
    e, p = cfg["empty"], cfg["wells"]["pitch_px_nominal"]
    w = max(3, int(round(e["window_frac"] * p)))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = cv2.blur(g, (w, w))
    std = np.sqrt(np.maximum(cv2.blur(g * g, (w, w)) - m * m, 0))
    sat = cv2.blur(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1], (w, w))
    return (std < e["max_std"]) & (sat < e["max_saturation"]) & (m > e["min_value"])


def empty_fraction(bgr: np.ndarray, pts: np.ndarray, cfg: dict) -> float:
    """Fraction of image points (e.g. the 96 predicted well centres) that land on visibly empty pixels.
    Points outside the image do not count as empty (they may simply be out of view)."""
    h, w = bgr.shape[:2]
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if not inside.any():
        return 0.0
    x0, y0 = np.floor(pts[inside].min(0)).astype(int) - 8
    x1, y1 = np.ceil(pts[inside].max(0)).astype(int) + 8
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    emp = empty_map(bgr[y0:y1, x0:x1], cfg)
    return float(np.sum(sample(emp, pts[inside], (x0, y0)))) / len(pts)


def well_area_polygon(quad: np.ndarray, cfg: dict) -> np.ndarray:
    """Image polygon of the well area inside a plate box (box outline minus the rim, plus a tolerance)."""
    from pipette_coach.localizer.geometry import project, quad_to_H

    m = cfg["wells"]["box_well_margin"]
    area = np.array([[-m, -m], [11 + m, -m], [11 + m, 7 + m], [-m, 7 + m]], np.float32)
    return project(quad_to_H(quad), area).astype(np.float32)


def in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    poly = poly.reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in pts], bool)


def plate_circles(bgr: np.ndarray, quad: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Standalone circle detection for one plate: radial-edge circles inside the box's well area.
    Returns (points, scores, n_removed_outside_box)."""
    import copy

    c = copy.deepcopy(cfg)
    c["wells"]["method"] = "radial_edge"
    area = well_area_polygon(quad, c)
    x0, y0 = np.floor(area.min(0)).astype(int) - 2
    x1, y1 = np.ceil(area.max(0)).astype(int) + 2
    P, S, _ = find_well_candidates(bgr, c, (int(x0), int(y0), int(x1), int(y1)))
    if len(P) == 0:
        return P, S, 0
    inside = in_polygon(P, area)
    return P[inside], S[inside], int((~inside).sum())
