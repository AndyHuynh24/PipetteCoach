"""Wells-first plate finding: scale and outline from the well lattice, the detector box only as a rough hint.

Why: a box detector trained on one camera can return HALF a plate (an upright plate seen as two landscape
blocks), which then gives a wrong scale and a wrong seed. The wells do not have that problem: the lattice of
well centres fixes the scale (its spacing), the orientation, and - when all 12 x 8 wells are visible - the plate
itself (outer wells + the measured rim), whatever the box said.

find_plate(frames): candidate scales from the detector boxes (box long side = 13.9 OR 9.5 spacings: whole plate
or half plate), wells detected by the well net in a wide region around every box, pooled over the frames,
lattice fitted; the scale whose lattice has most wells wins and is refined to the lattice's own spacing.
"""

import cv2
import numpy as np

from pipette_coach.localizer.detector import PlateDetector, boxes_of
from pipette_coach.localizer.plate_edges import measure_plate_box
from pipette_coach.localizer.plate_grid import SEARCH_SCALES, grow
from pipette_coach.localizer.refine import _largest_component
from pipette_coach.localizer.rings import in_polygon
from pipette_coach.localizer.wells_box import _lattice_model, fit_lattice, pool_points


def _boxes_any_scale(frame: np.ndarray, det: PlateDetector, conf: float) -> list[tuple[float, np.ndarray, float]]:
    """(search scale, box in ORIGINAL px, confidence) for the detector's best boxes at every search scale
    (conf >= conf). All scales are kept: the first scale that fires may have fired on another object (a tip
    box); the well lattice decides later which candidate is the plate."""
    out = []
    for s0 in SEARCH_SCALES:
        g = frame if s0 == 1.0 else cv2.resize(frame, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA if s0 < 1
                                               else cv2.INTER_LINEAR)  # fmt: skip
        r = det.model.predict(g, imgsz=det.imgsz, conf=conf, verbose=False, device=det.device)[0]
        quads, cf = boxes_of(r)  # oriented (OBB) or axis-aligned (detect model) boxes
        if len(cf):
            for k in np.argsort(-cf)[:3]:
                q = quads[k] / s0
                c, size = q.mean(0), np.ptp(q, axis=0).max()
                dup = any(np.linalg.norm(c - q2.mean(0)) < 0.3 * size and abs(np.ptp(q2, axis=0).max() / size - 1) < 0.3
                          for _, q2, _ in out)  # fmt: skip
                if not dup:  # the same object found at several scales is checked once
                    out.append((s0, q, float(cf[k])))
    return out


def _wells(frame_s: np.ndarray, region: np.ndarray, net) -> np.ndarray:
    x0, y0 = np.floor(region.min(0)).astype(int)
    x1, y1 = np.ceil(region.max(0)).astype(int)
    P, _ = net.detect(frame_s, (int(x0), int(y0), int(x1), int(y1)))
    return P[in_polygon(P, region.astype(np.float32))] if len(P) else P


def merge_close(P: np.ndarray, radius: float) -> np.ndarray:
    """One point per well: a camera that is not straight above a clear plate sees two rings per well (top opening
    and floor, offset by parallax), both detected; points within radius are merged (mean)."""
    used = np.zeros(len(P), bool)
    out = []
    for i in range(len(P)):
        if used[i]:
            continue
        m = (~used) & (np.linalg.norm(P - P[i], axis=1) < radius)
        used |= m
        out.append(P[m].mean(0))
    return np.array(out).reshape(-1, 2)


def _steps(H_rel: np.ndarray, ij: np.ndarray) -> np.ndarray:
    """Pixel length of one lattice step along u and v at the lattice centre."""
    c = ij.mean(0)
    p = cv2.perspectiveTransform(np.array([[c, c + [1, 0], c + [0, 1]]], np.float64), H_rel)[0]
    return np.array([np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0])])


def undouble(P: np.ndarray, lat, p_nom: float):
    """A camera that looks at a clear plate from the side sees each well's floor ring ~half a well beside its
    opening; both are detected and form a lattice with HALF the step along that axis (e.g. 16 instead of 8).
    Recognised by one step being ~half the other: keep one of the two interleaved sets (the one with more wells)
    and refit."""
    if lat is None:
        return None
    H_rel, ij, inl = lat
    st = _steps(H_rel, ij)
    if st.min() / st.max() > 0.7:
        return lat
    ax = int(np.argmin(st))
    idx = np.flatnonzero(inl)
    best = None
    for parity in (0, 1):
        keep = idx[(ij[:, ax] % 2) == parity]
        sub = fit_lattice(P[keep], (0.7 * p_nom, 1.4 * p_nom))
        if sub is not None and (best is None or len(sub[1]) > len(best[1])):
            mask = np.zeros(len(P), bool)
            mask[keep[sub[2]]] = True
            best = (sub[0], sub[1], mask)
    return best


def find_plate(
    frames: list,
    det: PlateDetector,
    net,
    cfg: dict,
    conf: float = 0.3,
    boxes: list | None = None,
    snap_edges: bool = True,
) -> dict | None:
    """Returns {"scale": s (resize so a spacing is ~pitch_px_nominal px), "H_rel", "ij" (lattice in the RESIZED
    frames), "wells" (pooled, resized px), "full": all 12 x 8 visible, "outline": plate quad in ORIGINAL px when
    full (else None)} or None."""
    p_nom = cfg["wells"]["pitch_px_nominal"]
    long_p, short_p = cfg["plate_edges"]["plate_size_pitch"]
    mid = frames[len(frames) // 2]
    if boxes is None:  # live use passes the box it already has (no second detector search)
        boxes = _boxes_any_scale(mid, det, conf)
    if not boxes:
        return None

    def attempt(q, s, cf):
        # wide region around this box: the whole plate fits even if the box is only half of it
        region = grow(cv2.boxPoints(cv2.minAreaRect((q * s).astype(np.float32))), p_nom, 6.0)
        pts = []
        for f in frames[:: max(1, len(frames) // 8)]:
            fs = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            pts.append(_wells(fs, region, net))
        P = merge_close(pool_points(pts, p_nom, min_frames=2 if len(pts) >= 4 else 1), 0.55 * p_nom)
        # the frame is scaled so a spacing is ~p_nom: only lattices of about that spacing count. Opening + floor
        # rings offset half a well diagonally also form a lattice, rotated 45 deg at ~0.71x the spacing
        lat = undouble(P, fit_lattice(P, (0.8 * p_nom, 1.25 * p_nom)), p_nom)
        # score: wells on the lattice, weighted by how sure the detector is that this is a plate (a tip box full
        # of tips can also form a 12 x 8 lattice)
        return None if lat is None else (len(lat[1]) * (0.5 + cf), s, lat, P)

    best = None
    for _, q, cf in boxes:
        long_px = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]))
        for n_pitch in (long_p, short_p):  # the box is the whole plate, or half of an upright one
            r = attempt(q, float(np.clip(p_nom / (long_px / n_pitch), 0.15, 3.0)), cf)
            if r is not None and (best is None or r[0] > best[0]):
                best = r + (q, cf)
    # few wells found: the scale guess from a box can be ~20 % off (a rotated plate's axis-aligned box is larger
    # than the plate) - try a little smaller and larger before giving up
    if best is None or len(best[2][1]) < 40:
        cands = [best] if best is not None else []
        for _, q, cf in boxes[:1] if best is None else [(None, best[4], best[5])]:
            s0 = best[1] if best is not None else float(np.clip(p_nom / (np.ptp(q, axis=0).max() / long_p), 0.15, 3))
            for k in (0.8, 1.25):
                r = attempt(q, float(np.clip(s0 * k, 0.15, 3.0)), cf)
                if r is not None:
                    cands.append(r + (q, cf))
        best = max(cands, key=lambda c: c[0]) if cands else None
    if best is not None:
        best = best[:4]
    if best is None:
        return None
    _, s, (H_rel, ij, inl), P = best
    # the plate = the largest connected block of lattice positions (one point each); stray detections elsewhere
    # that happen to land on far lattice positions must not stretch the span
    comp = _largest_component(ij)
    pts_in = P[inl][comp]  # inl: mask over P aligned with ij
    ij = ij[comp]
    if len(ij) >= 8:
        # model by how much of the plate is visible (similarity -> affine -> homography): a perspective fit to a
        # small patch of wells extrapolates wildly across the hidden part
        H_fit = _lattice_model(ij, pts_in, (7, 4))
        if H_fit is not None:
            H_rel = H_fit
    # refine the scale to the lattice's own spacing (mean of the two axis steps at the lattice centre)
    c = ij.mean(0)
    pts = cv2.perspectiveTransform(np.array([[c, c + [1, 0], c + [0, 1]]], np.float64), H_rel)[0]
    pitch = float((np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[0])) / 2)
    span = np.ptp(ij, axis=0)
    full = sorted(span.tolist()) == [7, 11]
    out = {"scale": s * p_nom / pitch, "scale_fit": s, "H_rel": H_rel, "ij": ij, "wells": P, "full": full,
           "outline": None}  # fmt: skip
    if full and snap_edges:
        lo, hi = ij.min(0), ij.max(0)
        u_long = span[0] == 11
        rim_u = ((long_p if u_long else short_p) - (11 if u_long else 7)) / 2
        rim_v = ((short_p if u_long else long_p) - (7 if u_long else 11)) / 2
        rect = np.array([[lo[0] - rim_u, lo[1] - rim_v], [hi[0] + rim_u, lo[1] - rim_v],
                         [hi[0] + rim_u, hi[1] + rim_v], [lo[0] - rim_u, hi[1] + rim_v]], np.float64)  # fmt: skip
        quad = cv2.perspectiveTransform(rect[None], H_rel)[0]
        # snap to the real edges (wells-derived quad as the seed; small search)
        fs = cv2.resize(mid, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        ecfg = {**cfg, "plate_edges": {**cfg["plate_edges"], "search_pitch": 0.6}}
        edge_q, scores, _ = measure_plate_box(fs, quad, pitch, ecfg)
        # trust the edge snap only where it stays close to the wells-derived outline (a wrong edge can pull a side)
        ok = np.linalg.norm(edge_q - quad, axis=1) < 0.5 * pitch
        out["outline"] = (np.where(ok[:, None], edge_q, quad) if (scores > 0).sum() >= 3 else quad) / s
        out["edge_sides"] = int((scores > 0).sum())
    return out


def plate_grid(r: dict, cfg: dict, rows_toward: str | None) -> np.ndarray:
    """12 x 8 grid (canonical -> ORIGINAL px) from a full-plate find_plate result: the lattice block is the plate,
    so its orientation and extent come from the wells, not from a detector box."""
    from pipette_coach.localizer.geometry import orient
    from pipette_coach.localizer.wells_box import grid_from_lattice

    long_p, short_p = cfg["plate_edges"]["plate_size_pitch"]
    ij = r["ij"]
    lo, hi = ij.min(0), ij.max(0)
    u_long = np.ptp(ij[:, 0]) == 11
    rim_u = ((long_p if u_long else short_p) - (11 if u_long else 7)) / 2
    rim_v = ((short_p if u_long else long_p) - (7 if u_long else 11)) / 2
    rect = (lo[0] - rim_u, hi[0] + rim_u, lo[1] - rim_v, hi[1] + rim_v)
    H = grid_from_lattice(r["H_rel"], ij, rect, bool(u_long), (12, 8))
    oc = cfg["orientation"]
    H = orient(H, oc["a1_rule"], oc["camera_mirrored"], rows_toward=rows_toward)
    return np.diag([1 / r["scale_fit"], 1 / r["scale_fit"], 1.0]) @ H


def trim_to_plate(ij: np.ndarray) -> np.ndarray:
    """A plate has at most 12 x 8 wells: while the lattice is longer than that along an axis (a sparse extra row of
    detections on the rim), drop the end row/column with fewer wells. The longer axis is trimmed to 12, the other
    to 8."""
    ij = np.asarray(ij)
    while True:
        n = np.ptp(ij, axis=0) + 1
        long_ax = int(np.argmax(n))
        cap = np.array([12, 8]) if long_ax == 0 else np.array([8, 12])
        over = [ax for ax in (0, 1) if n[ax] > cap[ax]]
        if not over:
            return ij
        ax = over[0]
        lo_n, hi_n = (ij[:, ax] == ij[:, ax].min()).sum(), (ij[:, ax] == ij[:, ax].max()).sum()
        ij = ij[ij[:, ax] != (ij[:, ax].min() if lo_n <= hi_n else ij[:, ax].max())]


def plausible_outline(quad: np.ndarray, vis_box: np.ndarray) -> bool:
    """A rebuilt plate outline must look like a plate seen from the camera: convex, corners between 55 and 125
    degrees (perspective allowed, not a skewed fan), long/short side ratio 1.1-2.2 (plate 1.46 + tilt), and not
    much bigger than the visible box (<= 2.6x its area: half the plate can be hidden, not more)."""
    q = np.asarray(quad, float)
    if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
        return False
    for k in range(4):
        a, b = q[k - 1] - q[k], q[(k + 1) % 4] - q[k]
        ang = np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1)))
        if not 55 <= ang <= 125:
            return False
    sides = np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)
    ratio = (sides[0] + sides[2]) / (sides[1] + sides[3])
    ratio = max(ratio, 1 / ratio)
    if not 1.1 <= ratio <= 2.2:
        return False
    area = cv2.contourArea(q.astype(np.float32))
    box_area = (vis_box[2] - vis_box[0]) * (vis_box[3] - vis_box[1])
    return area <= 2.6 * box_area


def rebuild_from_visible(r: dict, vis_box: np.ndarray, cfg: dict, rows_toward: str | None = None) -> dict | None:
    """Full plate from its VISIBLE part: a detector box around what is visible (ORIGINAL px, (x0, y0, x1, y1))
    plus the well lattice found inside it (find_plate result r).

    1. Length or width? Count visible wells along each lattice direction: more than 8 -> the 12-well (long) side.
       Exactly 8 with the plate's real edge at both ends (one rim beyond the outer wells) -> the complete 8-well
       (short) side, so the other direction is the long side. Anything else -> None (not guessed).
    2. Which end of a cut-off side is the real plate end: of the two possible placements, the one the visible box
       does not stick out of (the plate may continue under a hand past the box, not the other way round).
       Lattices longer than 12 x 8 (a sparse rim row) are trimmed first.
    3. Rebuild: the side is completed from its real end to 12 (or 8) wells + rims, in both directions (length
       from width and width from length), through the lattice homography (perspective-correct).
    Returns {"outline" (4, 2) ORIGINAL px, "H_grid" (canonical -> ORIGINAL px), "counts", "long_axis"} or None."""
    from pipette_coach.localizer.geometry import orient
    from pipette_coach.localizer.wells_box import grid_from_lattice

    long_p, short_p = cfg["plate_edges"]["plate_size_pitch"]
    rims = {12: (long_p - 11) / 2, 8: (short_p - 7) / 2}
    H, ij, s = r["H_rel"], r["ij"], r["scale_fit"]
    x0, y0, x1, y1 = np.asarray(vis_box, float) * s
    # wells can only be where the plate is visible: detections outside the visible-plate box (pipette tips,
    # fingers) that fall in line with the lattice would stretch it (13 "rows" on an 8-row plate)
    p_nom = cfg["wells"]["pitch_px_nominal"]
    xy = cv2.perspectiveTransform(ij[None].astype(np.float64), H)[0]
    m = 0.5 * p_nom
    inside = (xy[:, 0] > x0 - m) & (xy[:, 0] < x1 + m) & (xy[:, 1] > y0 - m) & (xy[:, 1] < y1 + m)
    if inside.sum() < 8:
        return None
    ij = trim_to_plate(ij[inside])
    lo, hi = ij.min(0), ij.max(0)
    n = hi - lo + 1
    corners = cv2.perspectiveTransform(np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float64),
                                       np.linalg.inv(H))[0]  # fmt: skip
    blo, bhi = corners.min(0), corners.max(0)  # visible box in lattice coords (pitch)
    gap_lo, gap_hi = lo - blo, bhi - hi  # visible extent beyond the outer wells
    is_end = 0.9  # a real plate end shows at least ~one rim (>= 1.27 pitch) beyond the outer wells; box slack
    if (n > 8).all():
        return None
    if n[0] > 8 or n[1] > 8:
        long_ax = 0 if n[0] > 8 else 1
        other = 1 - long_ax
        if n.max() >= 11:
            basis = "row12"
        elif n[other] == 8 and gap_lo[other] > is_end and gap_hi[other] > is_end:
            basis = "complete8"  # the other side is a complete 8-well column with the plate edge at both ends
        else:
            basis = "count>8"  # only 9-10 wells along a side: multichannel tips can add fake rows
    else:
        complete8 = [ax for ax in (0, 1) if n[ax] == 8 and gap_lo[ax] > is_end and gap_hi[ax] > is_end]
        if len(complete8) != 1:
            return None  # too little of the plate visible to tell length from width
        long_ax = 1 - complete8[0]
        basis = "complete8"
    rect = []
    for ax in (0, 1):
        N = 12 if ax == long_ax else 8
        rim = rims[N]
        if n[ax] == N:
            start = lo[ax]
        else:
            # the plate continues under a hand past the box, never the other way round: of the two placements
            # (anchored at the low or at the high wells), keep the one the visible box does not stick out of
            # and at the anchored (real) end the box edge lines up with the plate's edge
            cost = []
            for k, st in enumerate((lo[ax], hi[ax] - (N - 1))):
                p0, p1 = st - rim, st + N - 1 + rim
                sticks_out = max(0.0, p0 - blo[ax]) + max(0.0, bhi[ax] - p1)
                end_mismatch = abs(blo[ax] - p0) if k == 0 else abs(bhi[ax] - p1)
                cost.append(sticks_out + end_mismatch)
            if abs(cost[0] - cost[1]) < 0.5:
                return None  # both placements fit the box about equally: not guessed
            start = lo[ax] if cost[0] < cost[1] else hi[ax] - (N - 1)
        rect += [start - rim, start + N - 1 + rim]
    uL, uR, vT, vB = rect
    quad_s = cv2.perspectiveTransform(np.array([[[uL, vT], [uR, vT], [uR, vB], [uL, vB]]], np.float64), H)[0]
    # sanity: the visible box must lie (mostly) inside the rebuilt plate
    pm = np.zeros((int(max(y1, quad_s[:, 1].max())) + 2, int(max(x1, quad_s[:, 0].max())) + 2), np.uint8)
    cv2.fillConvexPoly(pm, np.round(quad_s).astype(np.int32), 1)
    vis = pm[max(int(y0), 0) : int(y1), max(int(x0), 0) : int(x1)]
    if vis.size == 0 or vis.mean() < 0.6:
        return None
    if not plausible_outline(quad_s, np.array([x0, y0, x1, y1])):
        return None
    Hg = grid_from_lattice(H, ij, (uL, uR, vT, vB), long_ax == 0, (12, 8))
    oc = cfg["orientation"]
    Hg = orient(Hg, oc["a1_rule"], oc["camera_mirrored"], rows_toward=rows_toward)
    S_inv = np.diag([1 / s, 1 / s, 1.0])
    return {
        "outline": quad_s / s,
        "H_grid": S_inv @ Hg,
        "counts": n.tolist(),
        "long_axis": long_ax,
        "basis": basis,
    }  # how length vs width was decided: row12 / complete8 (certain) or count>8 (weaker)


def plate_outline(H_grid: np.ndarray, cfg: dict) -> np.ndarray:
    """Plate outline (4, 2) in image px from a 12 x 8 grid (canonical -> image) and the plate's measured size in
    well spacings (outer wells + rims)."""
    long_p, short_p = cfg["plate_edges"]["plate_size_pitch"]
    ru, rv = (long_p - 11) / 2, (short_p - 7) / 2
    rect = np.array([[-ru, -rv], [11 + ru, -rv], [11 + ru, 7 + rv], [-ru, 7 + rv]], np.float64)
    return cv2.perspectiveTransform(rect[None], H_grid)[0]


def track_grid(
    H_prev: np.ndarray, dets: np.ndarray, cfg: dict, min_wells: int = 8, max_jump: float = 0.5
) -> tuple[np.ndarray, int] | None:
    """Follow a remembered 12 x 8 grid with the wells visible now. Memory gives every visible well its identity
    (row, column) - so which side of the plate is visible is known, however little is visible - and the grid is
    refitted to them (similarity / affine / homography by how much of the plate the wells span; the rest of the
    plate follows from the fit). Returns (H_grid, wells used) or None when too few wells (spread over < 2 rows or
    columns) are visible, or when a fit from few wells (< 40) would move the grid more than max_jump spacings
    (a real move with so little visible is not trusted)."""
    from pipette_coach.localizer.geometry import GRID, grid_error, pitch_px
    from pipette_coach.localizer.wells_box import snap_frame

    if len(dets) < min_wells:
        return None
    gate = cfg["plate_edges"]["well_gate_pitch"]
    H = H_prev
    n = 0
    for _ in range(2):
        _, matched, pos = snap_frame(H, dets, gate, 10**9)
        n = int(matched.sum())
        if n < min_wells or (np.ptp(GRID[matched], axis=0) < 1).any():
            return None
        H_new = _lattice_model(GRID[matched], pos[matched], (7, 4), ransac_px=0.15 * pitch_px(H))
        if H_new is None:
            return None
        H = H_new
    if n < 40 and grid_error(H_prev, H, ignore_labelling=False) > max_jump:
        return None
    return H, n
