"""Local grid refinement: snap a roughly-right grid onto visible wells.

snap_to_wells: from a detector box (could be ~0.4 pitch off, perspective unknown).
track:         from the previous frame's fit (fixed camera: perspective already right; only in-plane
               plate motion is estimated, so 6 visible wells anywhere on the plate are enough).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pipette_coach.localizer.geometry import GRID, grid_error, pitch_px, plate_bbox
from pipette_coach.localizer.rings import find_well_candidates


@dataclass
class RefineResult:
    H: np.ndarray
    n_matched: int
    ok: bool  # False -> caller should keep/discard the input grid (see status)
    status: str  # "refined" | "too_few" | "drifted"


def _match(H: np.ndarray, P: np.ndarray, gate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    proj = cv2.perspectiveTransform(GRID[None], H)[0]
    pitch = pitch_px(H)
    if len(P) == 0:
        return proj, np.zeros(len(GRID), int), np.zeros(len(GRID), bool), pitch
    d = np.linalg.norm(proj[:, None] - P[None], axis=2)
    j = d.argmin(1)
    return proj, j, d[np.arange(len(GRID)), j] < gate * pitch, pitch


def _candidates_near(bgr: np.ndarray, H: np.ndarray, cfg: dict, margin_pitch: float) -> np.ndarray:
    P, _, _ = find_well_candidates(bgr, cfg, roi=plate_bbox(H, margin_pitch))
    return P


def snap_to_wells(bgr: np.ndarray, H_init: np.ndarray, cfg: dict) -> RefineResult:
    """Snap a detector grid to ring candidates. Similarity fit (cannot squash an axis) unless most of
    the plate is visible, then a full homography. Refuses moves larger than snap.max_shift."""
    sc = cfg["snap"]
    P = _candidates_near(bgr, H_init, cfg, margin_pitch=1.0)
    H, n = np.asarray(H_init, np.float64), 0
    for _ in range(sc["rounds"]):
        _, j, ok, pitch = _match(H, P, sc["gate"])
        n = int(ok.sum())
        if n < sc["min_matches"]:
            return RefineResult(np.asarray(H_init, np.float64), n, False, "too_few")
        if n >= sc["full_homography_min"]:
            M, _ = cv2.findHomography(GRID[ok], P[j[ok]], cv2.RANSAC, 0.25 * pitch)
        else:
            S, _ = cv2.estimateAffinePartial2D(
                GRID[ok], P[j[ok]], method=cv2.RANSAC, ransacReprojThreshold=0.25 * pitch
            )
            M = None if S is None else np.vstack([S, [0, 0, 1]])
        if M is None:
            break
        H = M
    if grid_error(H, H_init) > sc["max_shift"]:
        return RefineResult(np.asarray(H_init, np.float64), n, False, "drifted")
    return RefineResult(H, n, True, "refined")


def track(bgr: np.ndarray, H_prev: np.ndarray, cfg: dict) -> RefineResult:
    """Update H_prev by the plate's in-plane motion (similarity delta), estimated from visible wells."""
    tc = cfg["track"]
    P = _candidates_near(bgr, H_prev, cfg, margin_pitch=tc["roi_margin_pitch"])
    H, n = np.asarray(H_prev, np.float64), 0
    for _ in range(tc["rounds"]):
        proj, j, ok, pitch = _match(H, P, tc["gate"])
        n = int(ok.sum())
        if n < tc["min_matches"]:
            return RefineResult(np.asarray(H_prev, np.float64), n, False, "too_few")
        S, _ = cv2.estimateAffinePartial2D(proj[ok], P[j[ok]], method=cv2.RANSAC, ransacReprojThreshold=0.2 * pitch)
        if S is None:
            return RefineResult(np.asarray(H_prev, np.float64), n, False, "too_few")
        H = np.vstack([S, [0, 0, 1]]) @ H
    if grid_error(H, H_prev, ignore_labelling=False) > tc["max_shift"]:
        return RefineResult(np.asarray(H_prev, np.float64), n, False, "drifted")
    # overall in-plane motion between the two grids must be physically plausible
    M, _ = cv2.estimateAffinePartial2D(
        cv2.perspectiveTransform(GRID[None], H_prev)[0], cv2.perspectiveTransform(GRID[None], H)[0]
    )
    rot = abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if rot > tc["max_rot_deg"] or abs(scale - 1) > tc["max_scale_change"]:
        return RefineResult(np.asarray(H_prev, np.float64), n, False, "implausible_motion")
    return RefineResult(H, n, True, "refined")


def _axis_vector(
    P: np.ndarray, nn: np.ndarray, axis: np.ndarray, cos_min: float, lo: float, hi: float
) -> np.ndarray | None:
    """Median nearest-neighbour vector along `axis` (sign-aligned), or None if too few."""
    v = (P[nn] - P[:, None]).reshape(-1, 2)
    n = np.linalg.norm(v, axis=1)
    keep = (n > lo) & (n < hi)
    v, n = v[keep], n[keep]
    c = v @ axis / n
    v = v[np.abs(c) > cos_min] * np.sign(c[np.abs(c) > cos_min])[:, None]
    return np.median(v, axis=0) if len(v) >= 5 else None


def _largest_component(ij: np.ndarray) -> np.ndarray:
    """Indices of the largest 4-connected cluster of lattice coords (1-node gaps allowed)."""
    S = {tuple(x): k for k, x in enumerate(ij)}
    seen, best = set(), []
    for st in S:
        if st in seen:
            continue
        comp, stack = [], [st]
        seen.add(st)
        while stack:
            a = stack.pop()
            comp.append(S[a])
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
                b = (a[0] + d[0], a[1] + d[1])
                if b in S and b not in seen:
                    seen.add(b)
                    stack.append(b)
        if len(comp) > len(best):
            best = comp
    return np.array(best, int)


def _col_axis_angle(H: np.ndarray) -> float:
    """Image angle (deg, mod 180) of the 12-column direction."""
    from pipette_coach.localizer.geometry import CENTER, project

    p = project(H, np.stack([CENTER, CENTER + [1, 0]]))
    d = p[1] - p[0]
    return float(np.degrees(np.arctan2(d[1], d[0])) % 180)


def snap_partial(
    bgr: np.ndarray, H_det: np.ndarray, quad: np.ndarray, cfg: dict, prior_H: np.ndarray | None = None
) -> RefineResult:
    """Place the full 12x8 grid using only the visible wells plus the detector box's reliable edges.

    1. Lattice (pitch, rotation) from neighbour vectors of ring candidates near the box.
    2. Every placement of the 12x8 window (both 12-along-u and 12-along-v) that covers all visible wells.
    3. Score = fraction of predicted wells inside the box that sit on rings
               - edge_weight * (box-vs-outline error on the best-matching side per axis).
    """
    pc, wc = cfg["partial_snap"], cfg["wells"]
    from pipette_coach.localizer.geometry import CENTER, OUTLINE, project  # local: avoid cycle at import
    from pipette_coach.localizer.rings import empty_fraction, ring_response, sample

    H_det = np.asarray(H_det, np.float64)
    roi = plate_bbox(H_det, pc["roi_margin_pitch"])
    P, _, _ = find_well_candidates(bgr, cfg, roi=roi)
    fail = RefineResult(H_det, len(P), False, "too_few")
    if len(P) < pc["min_wells"]:
        return fail
    lo, hi = wc["pitch_px_range"]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, 1e9)
    nn = np.argsort(D, 1)[:, :4]
    c0 = project(H_det, np.stack([CENTER, CENTER + [1, 0], CENTER + [0, 1]]))
    cos_min = np.cos(np.deg2rad(pc["axis_tol_deg"]))
    a_u = (c0[1] - c0[0]) / np.linalg.norm(c0[1] - c0[0])
    a_v = (c0[2] - c0[0]) / np.linalg.norm(c0[2] - c0[0])
    u = _axis_vector(P, nn, a_u, cos_min, lo, hi)
    v = _axis_vector(P, nn, a_v, cos_min, lo, hi)
    if u is None or v is None:
        return fail
    B = np.stack([u, v], 1)
    pitch = float((np.linalg.norm(u) + np.linalg.norm(v)) / 2)
    tol = pc["inlier_tol"] * pitch

    def lattice_inliers(origin: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cc = np.linalg.solve(basis, (P - origin).T).T
        rr = np.round(cc)
        return np.linalg.norm((cc - rr) @ basis.T, axis=1) < tol, rr

    # lattice phase: the origin candidate that most others agree with (a tip/edge blob would put it off-phase)
    p0 = max(P, key=lambda o: int(lattice_inliers(o, B)[0].sum()))
    inl, r = lattice_inliers(p0, B)
    # refine origin + basis by least squares on the inliers, then re-collect inliers
    A = np.hstack([r[inl], np.ones((int(inl.sum()), 1))])
    sol, *_ = np.linalg.lstsq(A, P[inl], rcond=None)
    B, p0 = sol[:2].T, sol[2]
    inl, r = lattice_inliers(p0, B)
    ij, pts = r[inl].astype(int), P[inl]
    if len(ij) < pc["min_wells"]:
        return fail
    k = _largest_component(ij)
    ij, pts = ij[k], pts[k]
    if len(ij) < pc["min_wells"]:
        return fail

    resp = ring_response(bgr[max(roi[1], 0) : roi[3], max(roi[0], 0) : roi[2]], cfg)
    off = (max(roi[0], 0), max(roi[1], 0))
    box_poly = np.asarray(quad, np.float32).reshape(-1, 1, 2)
    thr = cfg["grid_fit"]["verify_ring_score"]
    hyps = []
    for swap in (False, True):  # False: lattice u runs along the 12 columns
        nu, nv = (12, 8) if not swap else (8, 12)
        lo_u, hi_u = ij[:, 0].min(), ij[:, 0].max()
        lo_v, hi_v = ij[:, 1].min(), ij[:, 1].max()
        if hi_u - lo_u + 1 > nu or hi_v - lo_v + 1 > nv:
            continue
        for ou in range(hi_u - nu + 1, lo_u + 1):
            for ov in range(hi_v - nv + 1, lo_v + 1):
                local = (ij - [ou, ov]).astype(np.float32)
                canon = local if not swap else local[:, ::-1].copy()
                if len(canon) >= cfg["snap"]["full_homography_min"]:
                    H, _ = cv2.findHomography(canon, pts, cv2.RANSAC, 0.25 * pitch)
                else:
                    S, _ = cv2.estimateAffine2D(canon, pts, method=cv2.RANSAC, ransacReprojThreshold=0.25 * pitch)
                    H = None if S is None else np.vstack([S, [0, 0, 1]])
                if H is None:
                    continue
                if prior_H is not None:  # remembered plate orientation: 12 columns run the same way
                    dang = abs(_col_axis_angle(H) - _col_axis_angle(prior_H))
                    if min(dang, 180 - dang) > pc["prior_axis_tol_deg"]:
                        continue
                # wells the box says are visible should be on rings
                proj = project(H, GRID)
                inside = np.array(
                    [cv2.pointPolygonTest(box_poly, (float(x), float(y)), True) > 0.3 * pitch for x, y in proj]
                )
                ring_frac = float(np.mean(sample(resp, proj[inside], off) > thr)) if inside.any() else 0.0
                # box edges in this hypothesis' canonical frame vs the true plate outline
                qc = project(np.linalg.inv(H), np.asarray(quad, np.float32).reshape(4, 2))
                xs, ys = np.sort(qc[:, 0]), np.sort(qc[:, 1])
                ex = min(abs(xs[:2].mean() - OUTLINE[0, 0]), abs(xs[2:].mean() - OUTLINE[1, 0]))
                ey = min(abs(ys[:2].mean() - OUTLINE[0, 1]), abs(ys[2:].mean() - OUTLINE[2, 1]))
                emp = empty_fraction(bgr, proj, cfg)  # wells on bare bench -> misplaced
                score = ring_frac - pc["edge_weight"] * (ex + ey) - pc["empty_weight"] * emp
                hyps.append((score, ex + ey, H))
    if not hyps:
        return fail
    hyps.sort(key=lambda h: -h[0])
    best = hyps[0]
    if best[1] > pc["max_edge_err"]:
        return RefineResult(H_det, len(ij), False, "no_edge_match")
    if len(hyps) > 1 and best[0] - hyps[1][0] < pc["min_margin"]:
        return RefineResult(H_det, len(ij), False, "ambiguous")
    return RefineResult(best[2], len(ij), True, "partial")


@dataclass
class AlignResult:
    H: np.ndarray
    score: float  # mean radial-edge score at the 96 well positions (visible ones)
    margin: float  # best score minus best score >= `margin_dist` pitch away (half-offset guard)
    ok: bool


def _similarity_grid(t: np.ndarray, th_deg: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """All combinations as rows (tx, ty, theta_rad, scale)."""
    T = np.stack(np.meshgrid(t, t, np.radians(th_deg), sc, indexing="ij"), -1).reshape(-1, 4)
    return T


def align_grid(bgr: np.ndarray, H0: np.ndarray, cfg: dict, search: str = "detect") -> AlignResult:
    """Slide the whole 12x8 grid over the radial-edge score map and keep the in-plane motion (shift,
    rotation, scale about the plate centre) that maximises the summed score at the 96 well positions.

    One well's score barely separates a real opening from the gap between wells; summed over all visible
    wells it does, for any fill (the score is edge-based and polarity-free). Perspective comes from H0.
    """
    from pipette_coach.localizer.geometry import CENTER, project
    from pipette_coach.localizer.rings import radial_edge_response

    ac = cfg["align"]
    sc_cfg = ac[search]
    H0 = np.asarray(H0, np.float64)
    p = pitch_px(H0)
    x0, y0, x1, y1 = plate_bbox(H0, 1.0 + sc_cfg["shift"])
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    fail = AlignResult(H0, 0.0, 0.0, False)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return fail
    ratio, _ = radial_edge_response(bgr[y0:y1, x0:x1], cfg)
    P0 = project(H0, GRID).astype(np.float64) - [x0, y0]
    c0 = project(H0, CENTER[None]).astype(np.float64)[0] - [x0, y0]
    rel = P0 - c0

    def scores(params: np.ndarray) -> np.ndarray:
        tx, ty, th, s = (params[:, k][:, None] for k in range(4))
        cs, sn = np.cos(th) * s, np.sin(th) * s
        X = c0[0] + cs * rel[:, 0] - sn * rel[:, 1] + tx
        Y = c0[1] + sn * rel[:, 0] + cs * rel[:, 1] + ty
        inside = (X >= 0) & (X <= ratio.shape[1] - 1) & (Y >= 0) & (Y <= ratio.shape[0] - 1)
        v = cv2.remap(ratio, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
        n = np.maximum(inside.sum(1), 1)
        return np.where(inside.sum(1) >= ac["min_wells_in_view"], (v * inside).sum(1) / n, -1.0)

    coarse = _similarity_grid(
        np.arange(-sc_cfg["shift"], sc_cfg["shift"] + 1e-9, ac["coarse_step"]) * p,
        np.arange(-sc_cfg["rot_deg"], sc_cfg["rot_deg"] + 1e-9, sc_cfg["rot_step_deg"]),
        1 + np.arange(-sc_cfg["scale"], sc_cfg["scale"] + 1e-9, sc_cfg["scale_step"]),
    )
    sc_coarse = scores(coarse)
    b = coarse[int(np.argmax(sc_coarse))]
    fine = _similarity_grid(np.arange(-1, 1 + 1e-9, 0.2) * ac["coarse_step"] * p, [0.0], [1.0])
    fine[:, 0] += b[0]
    fine[:, 1] += b[1]
    fine = np.concatenate([fine[:, :2], np.tile(b[2:], (len(fine), 1))], 1)
    fine = np.concatenate(
        [fine + [0, 0, np.radians(dth), 0] for dth in np.arange(-1, 1.01, 0.5) * sc_cfg["rot_step_deg"] / 2]
        + [fine + [0, 0, 0, ds] for ds in (-0.5, 0.5) * np.array(sc_cfg["scale_step"])]
    )
    sc_fine = scores(fine)
    k = int(np.argmax(sc_fine))
    best, best_score = fine[k], float(sc_fine[k])
    far = np.hypot(coarse[:, 0] - best[0], coarse[:, 1] - best[1]) >= ac["margin_dist"] * p
    margin = best_score - (float(sc_coarse[far].max()) if far.any() else 0.0)
    tx, ty, th, s = best
    cs, sn = np.cos(th) * s, np.sin(th) * s
    cx, cy = c0 + [x0, y0]
    D = np.array([[cs, -sn, cx - cs * cx + sn * cy + tx], [sn, cs, cy - sn * cx - cs * cy + ty], [0, 0, 1]])
    H = D @ H0
    # whole-row ambiguity: the grid shifted by exactly one row/column must score clearly lower. With a row
    # hidden (e.g. under a multichannel pipette) "one row toward the rim" can score almost the same.
    shifted = [
        H @ np.array([[1, 0, dc], [0, 1, dr], [0, 0, 1]], np.float64) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
    ]

    def mean_score(Hs: np.ndarray) -> float:
        q = project(Hs, GRID).astype(np.float64) - [x0, y0]
        inside = (q[:, 0] >= 0) & (q[:, 0] <= ratio.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] <= ratio.shape[0] - 1)
        if inside.sum() < ac["min_wells_in_view"]:
            return -1.0
        v = cv2.remap(ratio, q[:, 0:1].astype(np.float32), q[:, 1:2].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
        return float((v * inside).sum() / inside.sum())

    lattice_margin = mean_score(H) - max(mean_score(Hs) for Hs in shifted)
    ok = best_score >= ac["min_score"] and margin >= ac["min_margin"] and lattice_margin >= ac["min_lattice_margin"]
    return AlignResult(H, best_score, min(margin, lattice_margin), ok)


def border_score(bgr: np.ndarray, H: np.ndarray, n: int = 40) -> np.ndarray:
    """Edge strength along each of the 4 sides of the plate outline predicted by H (grey-level step across
    the side, sampled at n points; points outside the image are ignored). Returns (4,) strengths."""
    from pipette_coach.localizer.geometry import OUTLINE, project

    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    q = project(H, OUTLINE).astype(np.float64)
    c = q.mean(0)
    h, w = g.shape
    out = np.zeros(4)
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        t = np.linspace(0.1, 0.9, n)[:, None]
        pts = a + t * (b - a)
        nrm = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        if np.dot(nrm, pts.mean(0) - c) < 0:
            nrm = -nrm  # outward
        inside_pts, outside_pts = pts - 2.5 * nrm, pts + 2.5 * nrm
        ok = (
            (np.minimum(inside_pts, outside_pts).min(1) >= 0) & (inside_pts[:, 0] < w - 1) & (outside_pts[:, 0] < w - 1)
        )
        ok &= (inside_pts[:, 1] < h - 1) & (outside_pts[:, 1] < h - 1)
        if ok.sum() < 5:
            continue
        vi = cv2.remap(
            g, inside_pts[ok, :1].astype(np.float32), inside_pts[ok, 1:].astype(np.float32), cv2.INTER_LINEAR
        )
        vo = cv2.remap(
            g, outside_pts[ok, :1].astype(np.float32), outside_pts[ok, 1:].astype(np.float32), cv2.INTER_LINEAR
        )
        out[k] = float(np.median(np.abs(vi - vo)))
    return out


def border_offsets(bgr: np.ndarray, H: np.ndarray, cfg: dict) -> list[tuple[int, float, float]]:
    """For each plate side: (side index, offset of the strongest straight edge from the predicted border in
    pitch units along the outward normal, peak contrast). The plate border is a continuous straight edge at a
    fixed distance from the outermost wells (SBS: 1.25 pitch long sides, 1.6 pitch short sides)."""
    from pipette_coach.localizer.geometry import OUTLINE, project

    bc = cfg["border"]
    x0, y0, x1, y1 = plate_bbox(H, bc["search"] + 1.0)
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    im = cv2.GaussianBlur(bgr[y0:y1, x0:x1], (0, 0), 1.0).astype(np.float32)
    mag = np.zeros(im.shape[:2], np.float32)
    for k in range(3):
        mag = np.maximum(
            mag, cv2.magnitude(cv2.Sobel(im[..., k], cv2.CV_32F, 1, 0), cv2.Sobel(im[..., k], cv2.CV_32F, 0, 1))
        )
    offs = np.arange(-bc["search"], bc["search"] + 1e-9, bc["step"])
    t = np.linspace(0.1, 0.9, bc["samples_along"])
    out = []
    for k in range(4):
        a, b = OUTLINE[k], OUTLINE[(k + 1) % 4]
        along = a[None] + t[:, None] * (b - a)[None]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([d[1], -d[0]], np.float32)  # outward for the clockwise canonical outline
        if np.dot(nrm, (a + b) / 2 - OUTLINE.mean(0)) < 0:
            nrm = -nrm
        pts = (along[None, :, :] + offs[:, None, None] * nrm[None, None, :]).reshape(-1, 2).astype(np.float32)
        q = project(H, pts) - [x0, y0]
        v = cv2.remap(mag, q[:, :1].astype(np.float32), q[:, 1:].astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
        prof = np.median(v.reshape(len(offs), len(t)), axis=1)  # median along the side: robust to partial cover
        j = int(np.argmax(prof))
        contrast = float(prof[j] / (np.median(prof) + 1e-6))
        out.append((k, float(offs[j]), contrast))
    return out
