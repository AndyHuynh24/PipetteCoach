"""Plate box reconstructed from the wells (lattice) + the plate's measured edges.

Clear plates make colour useless and a pipette can be parked over a side for a whole clip, so a side's edge is
not always measurable. The wells are: detected rings (hand-made clean-ring detector or the heatmap net) pooled
over a static window form a lattice; a homography H_rel maps relative lattice indices (u, v) -> image, which
handles perspective. In lattice coords the plate is the rectangle u in [uL, uR], v in [vT, vB]:
  - a side whose edge was clearly measured (plate_edges) is placed there (mapped into lattice coords), unless the
    pair's width disagrees with the plate size: then the side whose gap to the outermost wells is not a rim wide
    (a pipette boundary right next to the last visible wells) is treated as hidden;
  - a hidden side = the opposite measured side +- the plate's size in PITCH UNITS (measured on this plate's clean
    windows: scale-free, so it carries over between clips and camera distances);
  - both sides of an axis hidden: from the outer wells when the full 12 (or 8) are visible, + measured rim.
The box corners are the rectangle's corners projected through H_rel.
"""

import cv2
import numpy as np

from pipette_coach.localizer.plate_edges import box_sides


def pool_points(frames_pts: list[np.ndarray], pitch: float, min_frames: int = 2) -> np.ndarray:
    """Static plate: merge well detections over frames (same well within 0.3 pitch), keep wells seen in at least
    min_frames frames (one-off false detections on a hand drop out). Returns median positions (N, 2)."""
    pts = [p for p in frames_pts if len(p)]
    if not pts:
        return np.zeros((0, 2))
    allp = np.concatenate(pts)
    fid = np.concatenate([np.full(len(p), k) for k, p in enumerate(pts)])
    order = np.argsort(-np.bincount(fid)[fid])  # seed clusters from frames with many wells
    used = np.zeros(len(allp), bool)
    out = []
    for i in order:
        if used[i]:
            continue
        m = (~used) & (np.linalg.norm(allp - allp[i], axis=1) < 0.3 * pitch)
        used |= m
        if len(np.unique(fid[m])) >= min_frames:
            out.append(np.median(allp[m], axis=0))
    return np.array(out).reshape(-1, 2)


def fit_lattice(
    P: np.ndarray, pitch_range: tuple[float, float], seed: int = 0, iters: int = 300, min_span: tuple[int, int] = (7, 4)
):
    """Relative well lattice from points: RANSAC over nearest-neighbour bases, then homography refits.
    Returns (H_rel: (u, v) -> image, ij: (M, 2) int indices of inliers, inlier mask) or None.
    min_span: (long, short) lattice span needed for a perspective model; smaller -> affine."""
    if len(P) < 8:
        return None
    rng = np.random.default_rng(seed)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, 1e9)
    nn = np.argsort(D, 1)[:, :4]
    best = None
    for _ in range(iters):
        p = int(rng.integers(len(P)))
        u = P[nn[p, 0]] - P[p]
        lu = float(np.linalg.norm(u))
        if not pitch_range[0] < lu < pitch_range[1]:
            continue
        for q in nn[p, 1:]:
            v = P[q] - P[p]
            lv = float(np.linalg.norm(v))
            if 0.8 < lv / lu < 1.25 and abs(np.dot(u, v)) / (lu * lv) < 0.3:
                break
        else:
            continue
        B = np.stack([u, v], 1)
        c = np.linalg.solve(B, (P - P[p]).T).T
        r = np.round(c)
        inl = np.linalg.norm((c - r) @ B.T, axis=1) < 0.25 * lu
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), r, inl)
    if best is None or best[0] < 8:
        return None
    _, r, inl = best
    H, _ = cv2.findHomography(r[inl].astype(np.float32), P[inl].astype(np.float32), cv2.RANSAC, 2.0)
    if H is None:
        return None
    for _ in range(3):  # reassign every point to its nearest lattice node under the perspective model, refit
        Hi = np.linalg.inv(H)
        c = cv2.perspectiveTransform(P[None].astype(np.float64), Hi)[0]
        r = np.round(c)
        back = cv2.perspectiveTransform(r[None].astype(np.float64), H)[0]
        pitch = float(np.median(np.linalg.norm(np.diff(back[:2], axis=0)))) if len(back) > 1 else 14.0
        inl = np.linalg.norm(back - P, axis=1) < 0.25 * max(pitch, pitch_range[0])
        if inl.sum() < 8:
            return None
        H = _lattice_model(r[inl], P[inl], min_span)
    return H, r[inl].astype(int), inl


def _lattice_model(
    ij: np.ndarray, P: np.ndarray, min_span: tuple[int, int], ransac_px: float | None = None
) -> np.ndarray | None:
    """Model by how much of the plate the wells span: homography (perspective) when enough; affine when less;
    similarity when only a strip a few wells wide (extrapolating a richer model from a strip amplifies noise
    across the hidden part; top-view perspective is small and the well lattice is square).
    ransac_px: robust fit with this inlier threshold (else plain least squares / LMedS)."""
    span = sorted(np.ptp(ij, axis=0).tolist())[::-1]
    src, dst = ij.astype(np.float32), P.astype(np.float32)
    if span[0] >= min_span[0] and span[1] >= min_span[1]:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC if ransac_px else 0, ransac_px or 3.0)
        if H is not None:
            return H
    # a strip only a few wells wide: similarity (square lattice, top view) - the spacing across the strip then
    # comes from the long, well-measured direction instead of 1-2 spacings; otherwise affine
    fit = cv2.estimateAffinePartial2D if span[1] < min_span[1] else cv2.estimateAffine2D
    if ransac_px:
        A, _ = fit(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_px)
    else:
        A, _ = fit(src, dst, method=cv2.LMEDS)
    return None if A is None else np.vstack([A, [0, 0, 1]])


def _to_lat(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float64)[None], np.linalg.inv(H))[0]


def edge_sides_in_lattice(H: np.ndarray, edge_box: np.ndarray) -> dict[str, tuple[int, float]]:
    """Map the edge box's sides into lattice coords: {'uL'|'uR'|'vT'|'vB': (side index k, coordinate)}.
    The coordinate is taken where the side crosses the lattice centre line (sides are ~u=const or ~v=const)."""
    out = {}
    sides = box_sides(edge_box)
    c = _to_lat(H, edge_box.mean(0)[None])[0]
    for k, (a, b) in enumerate(sides):
        la, lb = _to_lat(H, np.array([a, b]))
        d = lb - la
        if abs(d[1]) >= abs(d[0]):  # runs along v -> it is a u = const side
            t = (c[1] - la[1]) / d[1] if d[1] else 0.5
            x = float(la[0] + t * d[0])
            out["uL" if x < c[0] else "uR"] = (k, x)
        else:
            t = (c[0] - la[0]) / d[0] if d[0] else 0.5
            y = float(la[1] + t * d[1])
            out["vT" if y < c[1] else "vB"] = (k, y)
    return out


def plate_size_pitch(
    H: np.ndarray, edge_box: np.ndarray, ij: np.ndarray, n_ls: tuple[int, int] = (12, 8), sym_tol: float = 0.3
) -> tuple[float, float]:
    """(long, short) plate size in pitch units from an edge box with all sides measured. An axis counts only when
    all its wells are visible AND the plate edges sit symmetrically around them (rim gaps equal within sym_tol):
    a pipette boundary taken for an edge, or hidden wells, fail that. Failed axes are NaN."""
    s = edge_sides_in_lattice(H, edge_box)
    if len(s) < 4:
        return float("nan"), float("nan")
    lo, hi = ij.min(0), ij.max(0)
    out = []
    for ax, (a, b) in enumerate((("uL", "uR"), ("vT", "vB"))):
        w = s[b][1] - s[a][1]
        n = round(hi[ax] - lo[ax]) + 1
        gap_a, gap_b = lo[ax] - s[a][1], s[b][1] - hi[ax]
        ok = n in n_ls and abs(gap_a - gap_b) < sym_tol
        out.append(w if ok else float("nan"))
    w, h = out
    if np.isnan(w) and np.isnan(h):
        return w, h
    if not np.isnan(w) and not np.isnan(h):
        return max(w, h), min(w, h)
    v = w if not np.isnan(w) else h
    return (v, float("nan")) if v > 11.5 else (float("nan"), v)  # long side > 11 pitch, short < 8 + rims


def box_from_wells(
    H: np.ndarray,
    ij: np.ndarray,
    edge_box: np.ndarray | None,
    edge_support: np.ndarray | None,
    size_ls: tuple[float, float] | None,
    n_ls: tuple[int, int],
    cfg: dict,
) -> tuple[np.ndarray, dict[str, str], dict] | None:
    """Plate box (4, 2) in the image + where each side came from ('edge' | 'size' | 'wells') + {u_long, rect: the
    box in lattice coords (uL, uR, vT, vB)}.
    size_ls = plate (long, short) size in pitch units; n_ls = wells along (long, short), e.g. (12, 8).
    Which lattice axis is the long one: from the edge box (aspect ~1.5 survives a hidden side), else the wells."""
    ec = cfg["plate_edges"]
    sides = edge_sides_in_lattice(H, edge_box) if edge_box is not None else {}
    lo, hi = ij.min(0), ij.max(0)
    if len(sides) == 4:
        u_long = sides["uR"][1] - sides["uL"][1] >= sides["vB"][1] - sides["vT"][1]
    else:
        u_long = hi[0] - lo[0] >= hi[1] - lo[1]
    size = None if size_ls is None else (size_ls if u_long else size_ls[::-1])
    n_wells = n_ls if u_long else n_ls[::-1]
    meas = {
        name: x for name, (k, x) in sides.items() if edge_support is not None and edge_support[k] >= ec["min_support"]
    }
    pos, src = {}, {}
    for ax, (a, b) in enumerate((("uL", "uR"), ("vT", "vB"))):
        W = size[ax] if size is not None else None
        full = hi[ax] - lo[ax] == n_wells[ax] - 1  # every row/column of wells along this axis is visible
        bad_width = W is not None and a in meas and b in meas and abs(meas[b] - meas[a] - W) > ec["size_tol_pitch"]
        if bad_width:
            # one "edge" is really a pipette/hand boundary seen all clip. A real plate edge sits one rim
            # beyond the outermost well; the side whose gap to the wells is further from the rim is the wrong one
            rim = (W - (n_wells[ax] - 1)) / 2
            gap_a, gap_b = lo[ax] - meas[a], meas[b] - hi[ax]
            if abs(gap_a - rim) > abs(gap_b - rim):
                meas.pop(a)
            else:
                meas.pop(b)
        if a in meas and b in meas:
            pos[a], pos[b], src[a], src[b] = meas[a], meas[b], "edge", "edge"
        elif a in meas and W is not None:
            pos[a], pos[b], src[a], src[b] = meas[a], meas[a] + W, "edge", "size"
        elif b in meas and W is not None:
            pos[a], pos[b], src[a], src[b] = meas[b] - W, meas[b], "size", "edge"
        elif full and W is not None:
            rim = (W - (n_wells[ax] - 1)) / 2  # outer wells + rim
            pos[a], pos[b], src[a], src[b] = lo[ax] - rim, hi[ax] + rim, "wells", "wells"
        else:
            return None
    rect = np.array([[pos["uL"], pos["vT"]], [pos["uR"], pos["vT"]], [pos["uR"], pos["vB"]], [pos["uL"], pos["vB"]]])
    info = {"u_long": bool(u_long), "rect": (pos["uL"], pos["uR"], pos["vT"], pos["vB"])}
    return cv2.perspectiveTransform(rect[None], H)[0], src, info


def grid_from_lattice(
    H_rel: np.ndarray, ij: np.ndarray, rect: tuple[float, float, float, float], u_long: bool, n_ls: tuple[int, int]
) -> np.ndarray:
    """Canonical plate grid (col 0..n_long-1, row 0..n_short-1) -> image, from the relative lattice. The window of
    n_long x n_short wells is centred in the plate box (wells sit centred on the plate), then nudged by whole wells
    only if that keeps more detected wells inside it (a rim measured a little off cannot shift the grid)."""
    uL, uR, vT, vB = rect
    n_u, n_v = n_ls if u_long else n_ls[::-1]
    starts = []
    for ax, (a, b, n) in enumerate(((uL, uR, n_u), (vT, vB, n_v))):
        centred = (a + b) / 2 - (n - 1) / 2
        best = None
        for s0 in range(int(np.floor(centred)) - 1, int(np.ceil(centred)) + 2):
            inside = int(((ij[:, ax] >= s0) & (ij[:, ax] <= s0 + n - 1)).sum())
            key = (inside, -abs(s0 - centred))
            if best is None or key > best[0]:
                best = (key, s0)
        starts.append(best[1])
    su, sv = starts
    # canonical (c, r) -> lattice (u, v): columns run along the long axis
    T = (
        np.array([[1, 0, su], [0, 1, sv], [0, 0, 1]], np.float64)
        if u_long
        else np.array([[0, 1, su], [1, 0, sv], [0, 0, 1]], np.float64)
    )
    return H_rel @ T


def snap_frame(
    H_grid: np.ndarray, dets: np.ndarray, gate: float = 0.3, min_matches: int = 12, min_span: tuple[int, int] = (7, 4)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One frame: match detections to grid nodes (each node <- nearest detection within gate pitch, one-to-one),
    refit the grid homography on the matches (RANSAC; lets the grid follow perspective / small moves), re-match.
    Detections on no node (pipette tips, a doubled column, hair) are dropped.
    Returns (H_frame, matched (96,) bool, positions (96, 2): the detection where matched, else the grid node)."""
    from pipette_coach.localizer.geometry import GRID, pitch_px, project

    H = H_grid
    for _ in range(2):
        g = project(H, GRID)
        p = pitch_px(H)
        matched = np.zeros(len(GRID), bool)
        pos = g.copy()
        if len(dets):
            d = np.linalg.norm(g[:, None] - dets[None], axis=2)
            for k in np.argsort(d.min(1)):  # greedy one-to-one, closest pairs first
                j = int(np.argmin(d[k]))
                if d[k, j] < gate * p:
                    matched[k], pos[k] = True, dets[j]
                    d[:, j] = np.inf
        if matched.sum() < min_matches:
            break
        Hn = _lattice_model(GRID[matched], pos[matched], min_span, ransac_px=0.15 * p)  # affine on a narrow strip
        if Hn is None:
            break
        H = Hn
    return H, matched, np.where(matched[:, None], pos, project(H, GRID))
