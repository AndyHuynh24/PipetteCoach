"""Global grid fit: find the 12x8 well lattice anywhere in the frame, without a detector.

Used to auto-label training frames for the plate detector and as a detector-free fallback.
Reliable when most of the plate is visible; it can lock onto other 8x12 grids (tip boxes) when the
plate is hidden, so callers must check `frac_on_rings` against grid_fit.confident_frac.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pipette_coach.localizer.geometry import GRID, N_COLS, N_ROWS
from pipette_coach.localizer.rings import find_well_candidates, sample


@dataclass
class GridFit:
    H: np.ndarray  # canonical -> image (labelling NOT yet oriented)
    frac_on_rings: float  # fraction of the 96 predicted centres sitting on a ring
    n_inliers: int


def _best_window(ij: np.ndarray) -> tuple[int, int, int, bool]:
    """Best-supported 12x8 window over integer lattice coords. Returns (count, i0, j0, swapped)."""
    best = (-1, 0, 0, False)
    if len(ij) == 0:
        return best
    mn = ij.min(0)
    g = ij - mn
    hist = np.zeros(g.max(0) + 1 + N_COLS, np.float32)
    np.add.at(hist, (g[:, 0], g[:, 1]), 1)
    S = np.pad(hist, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for swap, (a, b) in ((False, (N_COLS, N_ROWS)), (True, (N_ROWS, N_COLS))):
        A = S[a:, b:] - S[:-a, b:] - S[a:, :-b] + S[:-a, :-b]
        k = np.unravel_index(np.argmax(A), A.shape)
        if A[k] > best[0]:
            best = (int(A[k]), int(k[0] + mn[0]), int(k[1] + mn[1]), swap)
    return best


def _frac_on_rings(H: np.ndarray, resp: np.ndarray, thr: float) -> float:
    return float(np.mean(sample(resp, cv2.perspectiveTransform(GRID[None], H)[0]) > thr))


def _slide(H: np.ndarray, resp: np.ndarray) -> np.ndarray:
    """Shift the 12x8 window by +-1 column/row if that puts the centres on stronger rings (fixes
    windows that started one well off on a partly hidden plate)."""
    best = None
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            H2 = H @ np.array([[1, 0, dc], [0, 1, dr], [0, 0, 1]], np.float64)
            v = float(np.mean(np.clip(sample(resp, cv2.perspectiveTransform(GRID[None], H2)[0]), -30, 60)))
            if best is None or v > best[0]:
                best = (v, H2)
    return best[1]


def fit_grid(bgr: np.ndarray, cfg: dict, seed: int = 0) -> GridFit | None:
    """RANSAC over lattice bases from nearest-neighbour pairs, then 12x8 window, refit, and verify."""
    gc, wc = cfg["grid_fit"], cfg["wells"]
    rng = np.random.default_rng(seed)
    P, _, resp = find_well_candidates(bgr, cfg)
    if len(P) < 20:
        return None
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, 1e9)
    nn = np.argsort(D, 1)[:, :4]
    lo, hi = wc["pitch_px_range"]
    r0, r1 = gc["basis_len_ratio"]
    tol = gc["inlier_tol"]

    hyps = []
    for _ in range(gc["ransac_iters"]):
        p = int(rng.integers(len(P)))
        u = P[nn[p, 0]] - P[p]
        lu = float(np.linalg.norm(u))
        if not lo < lu < hi:
            continue
        vs = [P[q] - P[p] for q in nn[p, 1:]]
        vs = [
            v
            for v in vs
            if r0 < np.linalg.norm(v) / lu < r1 and abs(np.dot(u, v)) / (lu * np.linalg.norm(v)) < gc["basis_max_cos"]
        ]
        if not vs:
            continue
        B = np.stack([u, vs[0]], 1)
        c = np.linalg.solve(B, (P - P[p]).T).T
        r = np.round(c)
        inl = np.linalg.norm((c - r) @ B.T, axis=1) < tol * lu
        cnt, i0, j0, swap = _best_window(r[inl].astype(int))
        hyps.append((cnt, p, B, i0, j0, swap))

    best: GridFit | None = None
    for _, p, B, i0, j0, swap in sorted(hyps, key=lambda h: -h[0])[: gc["verify_top_k"]]:
        c = np.linalg.solve(B, (P - P[p]).T).T
        r = np.round(c)
        res = np.linalg.norm((c - r) @ B.T, axis=1)
        a, b = (N_COLS, N_ROWS) if not swap else (N_ROWS, N_COLS)
        m = (res < tol * np.linalg.norm(B, axis=0).min()) & (r[:, 0] >= i0) & (r[:, 0] < i0 + a)
        m &= (r[:, 1] >= j0) & (r[:, 1] < j0 + b)
        if m.sum() < 12:
            continue
        ij = r[m] - [i0, j0]
        canon = (ij if not swap else ij[:, ::-1]).astype(np.float32)
        H, _ = cv2.findHomography(canon, P[m], cv2.RANSAC, 3.0)
        if H is None:
            continue
        n_ok = 0
        for _ in range(3):
            proj = cv2.perspectiveTransform(GRID[None], H)[0]
            pitch = np.linalg.norm(proj[1] - proj[0])
            d = np.linalg.norm(proj[:, None] - P[None], axis=2)
            j = d.argmin(1)
            ok = d[np.arange(len(GRID)), j] < 0.3 * pitch
            if ok.sum() < 8:
                break
            H, _ = cv2.findHomography(GRID[ok], P[j[ok]], 0)
            H = _slide(H, resp)
            n_ok = int(ok.sum())
        frac = _frac_on_rings(H, resp, gc["verify_ring_score"])
        if best is None or frac > best.frac_on_rings:
            best = GridFit(H=H, frac_on_rings=frac, n_inliers=n_ok)
    return best
