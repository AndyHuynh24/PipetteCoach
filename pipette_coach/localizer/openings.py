"""Sub-pixel well-opening refinement from the circular edge of each well.

The ring filter peaks on the dark well bottom. Seen from a slight angle the bottom shifts relative to
the opening, and filled wells may have no dark bottom at all. The opening's outer edge (bright wall ->
plate) is visible either way, so each well's opening circle is fitted from it and the plate grid is
refit on those centres.
"""

import cv2
import numpy as np

from pipette_coach.localizer.geometry import GRID, grid_error, pitch_px, project


def well_patches(bgr: np.ndarray, centres: np.ndarray, p: float, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Grey patches (N, n, n) centred (sub-pixel) on the given image points; t = sample offsets in px."""
    oc = cfg["openings"]
    n, half = oc["patch_px"], oc["patch_half"] * p
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.5)
    t = np.linspace(-half, half, n)
    X, Y = np.meshgrid(t, t)
    mx = (centres[:, 0][:, None, None] + X[None]).reshape(-1, n).astype(np.float32)
    my = (centres[:, 1][:, None, None] + Y[None]).reshape(-1, n).astype(np.float32)
    return cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(len(centres), n, n), t


def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    A = np.stack([x, y, np.ones_like(x)], 1)
    (a, b, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = a / 2, b / 2
    return float(cx), float(cy), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def opening_offset(patch: np.ndarray, t: np.ndarray, p: float, cfg: dict) -> np.ndarray | None:
    """Opening centre relative to the patch centre (px), or None if the edge is not clear.

    Per ray: the outer edge is where brightness falls half-way from the wall peak to the plate level
    outside it (sub-pixel). A circle is fitted to the edge points with iterative outlier rejection.
    """
    oc = cfg["openings"]
    n, step = len(t), t[1] - t[0]
    c = (n - 1) / 2
    th = np.linspace(0, 2 * np.pi, oc["n_rays"], endpoint=False)
    rr = np.linspace(0.1 * p, 0.6 * p, 60)
    mx = (c + np.cos(th)[:, None] * rr[None] / step).astype(np.float32)
    my = (c + np.sin(th)[:, None] * rr[None] / step).astype(np.float32)
    prof = cv2.remap(patch.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    xs, ys = [], []
    lim = int(0.75 * len(rr))
    for k in range(len(th)):
        pr = prof[k]
        ip = int(np.argmax(pr[:lim]))  # wall peak
        outside = pr[ip:].min()
        half = (pr[ip] + outside) / 2
        if pr[ip] - outside < oc["min_contrast"]:
            continue
        j = ip + int(np.argmax(pr[ip:] < half))
        if j <= ip:
            continue
        r = rr[j - 1] + (pr[j - 1] - half) / (pr[j - 1] - pr[j] + 1e-6) * (rr[j] - rr[j - 1])
        xs.append(r * np.cos(th[k]))
        ys.append(r * np.sin(th[k]))
    if len(xs) < oc["min_ray_frac"] * len(th):
        return None
    x, y = np.array(xs), np.array(ys)
    keep = np.ones(len(x), bool)
    for _ in range(3):
        cx, cy, R = _fit_circle(x[keep], y[keep])
        res = np.abs(np.hypot(x - cx, y - cy) - R)
        keep = res < max(0.6, 2.5 * float(np.median(res)))
        if keep.sum() < oc["min_ray_frac"] * len(th):
            return None
    if not 0.28 * p < R < 0.52 * p:
        return None
    return np.array([cx, cy])


def opening_centres(bgr: np.ndarray, H: np.ndarray, cfg: dict) -> np.ndarray:
    """(96, 2) opening centres in image px (NaN where the edge was not clear or too far from the grid)."""
    pr = project(H, GRID).astype(np.float64)
    p = pitch_px(H)
    h, w = bgr.shape[:2]
    out = np.full((len(GRID), 2), np.nan)
    inside = (pr[:, 0] > 0.6 * p) & (pr[:, 0] < w - 0.6 * p) & (pr[:, 1] > 0.6 * p) & (pr[:, 1] < h - 0.6 * p)
    if not inside.any():
        return out
    idx = np.nonzero(inside)[0]
    patches, t = well_patches(bgr, pr[idx], p, cfg)
    for k, patch in zip(idx, patches, strict=True):
        o = opening_offset(patch, t, p, cfg)
        if o is not None and np.linalg.norm(o) < cfg["openings"]["max_offset"] * p:
            out[k] = pr[k] + o
    return out


def refine_to_openings(bgr: np.ndarray, H: np.ndarray, cfg: dict) -> tuple[np.ndarray, int]:
    """Refit the grid on the wells' opening centres. Returns (H, n_wells_used); input H if not enough
    clear wells or the refit would move the grid more than openings.max_change."""
    oc = cfg["openings"]
    C = opening_centres(bgr, H, cfg)
    ok = np.isfinite(C[:, 0])
    if ok.sum() < oc["min_wells"]:
        return H, int(ok.sum())
    p = pitch_px(H)
    Hn, inl = cv2.findHomography(GRID[ok], C[ok].astype(np.float32), cv2.RANSAC, oc["ransac_frac"] * p)
    if Hn is None or grid_error(Hn, H, ignore_labelling=False) > oc["max_change"]:
        return H, int(ok.sum())
    return Hn, int(inl.sum())


def ring_edge_support(bgr: np.ndarray, H: np.ndarray, cfg: dict) -> int:
    """Number of predicted wells around which a clean ring-edge circle is found CENTRED on the prediction.

    The ring's outer edge (well wall -> plate) is clear for empty and filled wells alike. On a correct grid most
    visible wells give a centred circle; half a well off, the predictions sit in the gaps between wells and no
    centred circle fits there."""
    C = opening_centres(bgr, H, cfg)
    ok = np.isfinite(C[:, 0])
    if not ok.any():
        return 0
    d = np.linalg.norm(C[ok] - project(H, GRID)[ok], axis=1)
    return int((d < cfg["ring_edge"]["centred_frac"] * pitch_px(H)).sum())


def ring_edge_placement(bgr: np.ndarray, H: np.ndarray, cfg: dict) -> tuple[np.ndarray, int, int]:
    """Best of H and its half-well / one-row / one-column shifts by ring-edge support.
    Returns (H_best, support, margin over the runner-up)."""
    from pipette_coach.localizer.verifier import OFFSETS, offset_H  # numpy-only helpers (torch is lazy there)

    cands = [offset_H(H, dx, dy) for dx, dy in OFFSETS]
    sup = np.array([ring_edge_support(bgr, Hc, cfg) for Hc in cands])
    order = np.argsort(-sup)
    return cands[order[0]], int(sup[order[0]]), int(sup[order[0]] - sup[order[1]])


def _fit_quality(patch: np.ndarray, t: np.ndarray, p: float, cfg: dict) -> tuple[np.ndarray, float, float] | None:
    """Like opening_offset, but also returns radius, residual, share of rays with an edge, and radius spread
    (pitch units) so clean rings can be told from the gaps between wells."""
    oc = cfg["openings"]
    n, step = len(t), t[1] - t[0]
    c = (n - 1) / 2
    th = np.linspace(0, 2 * np.pi, oc["n_rays"], endpoint=False)
    rr = np.linspace(0.1 * p, 0.6 * p, 60)
    mx = (c + np.cos(th)[:, None] * rr[None] / step).astype(np.float32)
    my = (c + np.sin(th)[:, None] * rr[None] / step).astype(np.float32)
    prof = cv2.remap(patch.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    xs, ys = [], []
    lim = int(0.75 * len(rr))
    for k in range(len(th)):
        pr = prof[k]
        ip = int(np.argmax(pr[:lim]))
        outside = pr[ip:].min()
        half = (pr[ip] + outside) / 2
        if pr[ip] - outside < oc["min_contrast"]:
            continue
        j = ip + int(np.argmax(pr[ip:] < half))
        if j <= ip:
            continue
        r = rr[j - 1] + (pr[j - 1] - half) / (pr[j - 1] - pr[j] + 1e-6) * (rr[j] - rr[j - 1])
        xs.append(r * np.cos(th[k]))
        ys.append(r * np.sin(th[k]))
    if len(xs) < oc["min_ray_frac"] * len(th):
        return None
    x, y = np.array(xs), np.array(ys)
    ray_frac = len(xs) / len(th)
    radius_iqr = float(np.subtract(*np.percentile(np.hypot(x, y), [75, 25]))) / p
    keep = np.ones(len(x), bool)
    for _ in range(3):
        cx, cy, R = _fit_circle(x[keep], y[keep])
        res = np.abs(np.hypot(x - cx, y - cy) - R)
        keep = res < max(0.6, 2.5 * float(np.median(res)))
        if keep.sum() < oc["min_ray_frac"] * len(th):
            return None
    return np.array([cx, cy]), R, float(np.median(res[keep])), ray_frac, radius_iqr


def detect_rings(bgr: np.ndarray, quad: np.ndarray, cfg: dict) -> np.ndarray:
    """Bottom-up ring detection inside the plate box: circle-fit the ring edge around every candidate spot and keep
    only clean circles (right radius, low residual, enough edge rays, centred near the probe). The gaps between
    wells are bounded by four concave edges - a diamond, not a circle - so they fail the fit. Returns (N, 2)."""
    from pipette_coach.localizer.rings import find_well_candidates, in_polygon, plate_circles, well_area_polygon

    rc = cfg["ring_edge"]
    p = cfg["wells"]["pitch_px_nominal"]
    area = well_area_polygon(quad, cfg)
    x0, y0 = np.floor(area.min(0)).astype(int) - 2
    x1, y1 = np.ceil(area.max(0)).astype(int) + 2
    import copy

    rcfg = copy.deepcopy(cfg)
    rcfg["wells"]["method"] = "ring"
    R, _, _ = find_well_candidates(bgr, rcfg, (int(x0), int(y0), int(x1), int(y1)))
    E, _, _ = plate_circles(bgr, quad, cfg)
    probes = np.concatenate([a for a in (R, E) if len(a)]) if (len(R) or len(E)) else np.zeros((0, 2))
    if len(probes) == 0:
        return probes
    probes = probes[in_polygon(probes, area)]
    patches, t = well_patches(bgr, probes, p, cfg)
    rings = []
    for q, patch in zip(probes, patches, strict=True):
        r = _fit_quality(patch, t, p, cfg)
        if r is None:
            continue
        off, rad, res, ray_frac, radius_iqr = r
        if (
            np.linalg.norm(off) < rc["probe_max_offset"] * p
            and rc["radius_frac"][0] * p < rad < rc["radius_frac"][1] * p
            and res < rc["max_residual_px"]
            and ray_frac >= rc["min_ray_frac"]  # a real ring has an edge all the way around ...
            and radius_iqr <= rc["max_radius_iqr"]  # ... at the same radius; gaps have partial, uneven edges
        ):
            rings.append(q + off)
    if not rings:
        return np.zeros((0, 2))
    rings = np.array(rings)
    # merge duplicates (ring + edge probes on the same well)
    keep: list[np.ndarray] = []
    for r in rings:
        if all(np.linalg.norm(r - k) > 0.4 * p for k in keep):
            keep.append(r)
    return np.array(keep)


def reconstruct_from_rings(rings: np.ndarray, H_guess: np.ndarray, cfg: dict) -> tuple[np.ndarray, int] | None:
    """Reconstruct the full plate (all 96 wells) backward from detected rings: assign each ring to its nearest
    well of the guessed grid, then fit a homography (RANSAC) canonical -> ring centres. The guess only has to be
    within ~1/3 pitch; whole-row ambiguity is handled before this (verifier / ring-edge placement)."""
    if len(rings) < cfg["ring_edge"]["min_rings"]:
        return None
    p = pitch_px(H_guess)
    g = project(H_guess, GRID)
    d = np.linalg.norm(g[:, None] - rings[None], axis=2)
    j = d.argmin(1)
    ok = d[np.arange(len(GRID)), j] < 0.35 * p
    if ok.sum() < cfg["ring_edge"]["min_rings"]:
        return None
    H, inl = cv2.findHomography(GRID[ok], rings[j[ok]].astype(np.float32), cv2.RANSAC, 0.1 * p)
    if H is None:
        return None
    return H, int(inl.sum())


def ring_placement(rings: np.ndarray, H: np.ndarray, gate: float = 0.25) -> tuple[np.ndarray, int, int]:
    """Best of H and its half-well / one-row / one-column shifts by how many detected rings land on its wells.
    Returns (H_best, rings on wells, margin over the runner-up)."""
    from pipette_coach.localizer.verifier import OFFSETS, offset_H

    cands = [offset_H(H, dx, dy) for dx, dy in OFFSETS]
    if len(rings) == 0:
        return H, 0, 0
    counts = []
    for Hc in cands:
        g = project(Hc, GRID)
        counts.append(int((np.linalg.norm(g[:, None] - rings[None], axis=2).min(1) < gate * pitch_px(Hc)).sum()))
    order = np.argsort(counts)[::-1]
    return cands[order[0]], counts[order[0]], counts[order[0]] - counts[order[1]]
