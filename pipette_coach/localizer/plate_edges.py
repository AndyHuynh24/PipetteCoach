"""Measure the plate's real outer edges (instead of drawing the detector box or a standard outline).

Each side of the plate is a long straight edge with the SAME polarity along its whole length (plate on one side,
bench on the other). Well rings near a side alternate polarity and leave gaps between wells, so they score low.
Per side: sample the signed colour gradient along the outward normal at points along the rough side (from the
detector box), then search lines with independent end offsets (covers position and tilt) for the largest |median|
signed gradient. The median ignores a hand or pipette over less than half the side. Of several strong parallel
edges, colour vetoes those with clearly non-bench colour right outside (pipette/hand boundary), and the outermost
remaining one is the plate's footprint (not the inner wall). Sides are then intersected.
"""

import cv2
import numpy as np


def _normal_gradient(bgr: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.GaussianBlur(bgr.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3).sum(2) / 3
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3).sum(2) / 3
    return gx, gy


def _colour_mode(lab_px: np.ndarray, bin_size: float) -> np.ndarray:
    """Most common colour (Lab) of a set of pixels: hands, hair or a pipette near the plate are not the mode."""
    q = (lab_px // bin_size).astype(np.int32)
    _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    return lab_px[inv.ravel() == np.argmax(cnt)].mean(0)


def bench_colour(lab: np.ndarray, quad: np.ndarray, pitch: float, cfg: dict) -> np.ndarray | None:
    """Bench colour = mode of a band around the (rough) plate box."""
    ec = cfg["plate_edges"]
    q = np.asarray(quad, np.float64)
    c = q.mean(0)

    def grow(k: float) -> np.ndarray:
        return c + (q - c) * (1 + k * pitch / np.linalg.norm(q - c, axis=1, keepdims=True))

    band = np.zeros(lab.shape[:2], np.uint8)
    cv2.fillConvexPoly(band, np.round(grow(ec["bench_band_pitch"][1])).astype(np.int32), 1)
    cv2.fillConvexPoly(band, np.round(grow(ec["bench_band_pitch"][0])).astype(np.int32), 0)
    if band.sum() < 50:
        return None
    return _colour_mode(lab[band > 0], ec["bench_bin"])


def _outside_colour(lab, bench, a, d, n, t, s_a, s_b, delta) -> float:
    """Median colour distance to the bench just OUTSIDE the line. Measured: real plate edges median ~27 (p90 ~65),
    edges that are really a pipette boundary ~80-90. The inner wall vs the outer edge can NOT be told apart by
    colour (clear plastic: both look like the bench) - the outermost rule does that."""
    s_t = s_a + t * (s_b - s_a)
    P = a[None] + t[:, None] * d[None] + (s_t[:, None] + delta) * n[None]
    v = cv2.remap(lab, P[:, 0:1].astype(np.float32), P[:, 1:2].astype(np.float32), cv2.INTER_LINEAR)[:, 0]
    return float(np.median(np.linalg.norm(v - bench, axis=1)))


def _fit_side(
    gx, gy, a, b, centre, search, cfg, tilt: tuple[float, float] | None = None, lab=None, bench=None
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Best straight edge near the side a-b. Returns (new a, new b, score = |median signed gradient|).
    tilt = (target, tol) radians: only lines whose tilt relative to a-b is within tol of target (parallel fit)."""
    ec = cfg["plate_edges"]
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1:
        return None
    n = np.array([-d[1], d[0]]) / L
    if np.dot((a + b) / 2 - centre, n) < 0:
        n = -n  # outward
    t = np.linspace(ec["end_trim"], 1 - ec["end_trim"], ec["n_samples"])
    s = np.arange(-search, search + 1e-6, ec["step_px"])
    P = a[None, None] + t[:, None, None] * d[None, None] + s[None, :, None] * n[None, None]  # (T, S, 2)
    mx, my = P[..., 0].astype(np.float32), P[..., 1].astype(np.float32)
    G = cv2.remap(gx, mx, my, cv2.INTER_LINEAR, borderValue=np.nan) * n[0]
    G = G + cv2.remap(gy, mx, my, cv2.INTER_LINEAR, borderValue=np.nan) * n[1]  # (T, S) signed normal gradient
    S = len(s)
    ia, ib = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    # offset index along the line for every (end a offset, end b offset) pair
    idx = np.rint(ia[..., None] + t[None, None] * (ib - ia)[..., None]).astype(int)  # (S, S, T)
    vals = G[np.arange(len(t))[None, None], idx]  # (S, S, T)
    with np.errstate(all="ignore"):
        med = np.nanmedian(vals, axis=2)
    if np.all(np.isnan(med)):
        return None
    score = np.abs(np.nan_to_num(med)).astype(np.float32)
    if tilt is not None:
        dtilt = np.arctan2(s[None, :] - s[:, None], L)  # tilt of the line given end offsets (i, j)
        score[np.abs(dtilt - tilt[0]) > tilt[1]] = 0.0
        if score.max() <= 0:
            return None
    # the plate has more than one straight edge per side (raised wall around the wells, outer footprint edge, a
    # pipette or hand boundary, a shadow line): candidates = strong local peaks (>= rel_score of the best)
    peak = (score == cv2.dilate(score, np.ones((5, 5), np.uint8))) & (score >= ec["rel_score"] * score.max())
    ii, jj = np.nonzero(peak)
    if lab is not None and bench is not None:
        # colour veto: an edge with clearly non-bench colour right outside it is a pipette/hand boundary
        p = search / ec["search_pitch"]
        out = np.array([_outside_colour(lab, bench, a, d, n, t, s[i], s[j], ec["colour_delta_pitch"] * p)
                        for i, j in zip(ii, jj, strict=True)])  # fmt: skip
        ok = out <= ec["colour_veto_dist"]
        if not ok.any():
            return None  # every strong edge here is a pipette/hand boundary: side hidden in this frame
        ii, jj = ii[ok], jj[ok]
    k = int(np.argmax(s[ii] + s[jj]))  # outermost strong edge = the plate's footprint (not the inner wall)
    i, j = ii[k], jj[k]
    return a + s[i] * n, b + s[j] * n, float(score[i, j])


def _intersect(p1, p2, q1, q2) -> np.ndarray:
    A = np.array([p2 - p1, q1 - q2]).T
    u = np.linalg.solve(A, q1 - p1)
    return p1 + u[0] * (p2 - p1)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


def _wrap(x: float) -> float:
    return float((x + np.pi / 2) % np.pi - np.pi / 2)  # line angle difference in (-90, 90] deg


def measure_plate_box(
    bgr: np.ndarray, quad: np.ndarray, pitch: float, cfg: dict
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Plate outline (4, 2) from the real edges, seeded by a rough quad (detector box).
    Returns (quad, per-side scores, per-side lines (a, b)). A side whose edge is too weak (hidden) keeps the rough
    side and gets score 0. Opposite sides must be parallel (within parallel_tol_deg): when they are not, the weaker
    side is re-searched among lines parallel to the stronger one."""
    ec = cfg["plate_edges"]
    gx, gy = _normal_gradient(bgr, ec["sigma_px"])
    q = np.asarray(quad, np.float64)
    centre = q.mean(0)
    search = ec["search_pitch"] * pitch
    lab = bench = None
    if ec["use_colour"]:
        lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), ec["sigma_px"]), cv2.COLOR_BGR2LAB).astype(np.float32)
        bench = bench_colour(lab, q, pitch, cfg)
    lines, scores = [], []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        r = _fit_side(gx, gy, a, b, centre, search, cfg, lab=lab, bench=bench)
        if r is None or r[2] < ec["min_score"]:
            lines.append((a, b))
            scores.append(0.0)
        else:
            lines.append((r[0], r[1]))
            scores.append(r[2])
    tol = np.deg2rad(ec["parallel_tol_deg"])
    for k in range(2):  # opposite pairs (0, 2), (1, 3)
        i, j = k, k + 2
        if scores[i] == 0 and scores[j] == 0:
            continue
        diff = _wrap(_angle(*lines[i]) - _angle(*lines[j]))
        if abs(diff) <= tol and scores[i] > 0 and scores[j] > 0:
            continue
        strong, weak = (i, j) if scores[i] >= scores[j] else (j, i)
        a, b = q[weak], q[(weak + 1) % 4]
        target = _wrap(_angle(*lines[strong]) - _angle(a, b))
        tilt = (target, np.deg2rad(ec["parallel_fit_deg"]))
        r = _fit_side(gx, gy, a, b, centre, search, cfg, tilt=tilt, lab=lab, bench=bench)
        if r is not None and r[2] >= ec["min_score"]:
            lines[weak], scores[weak] = (r[0], r[1]), r[2]
        else:
            lines[weak], scores[weak] = (a, b), 0.0
    out = np.array([_intersect(*lines[k - 1], *lines[k]) for k in range(4)])
    return out, np.array(scores), lines


def _hesse(a: np.ndarray, b: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """Line through a, b as (normal angle, distance) with the normal pointing away from ref."""
    d = b - a
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    if np.dot(a - ref, n) < 0:
        n = -n
    return float(np.arctan2(n[1], n[0])), float(np.dot(a, n))


def fuse_boxes(measured: list[tuple[np.ndarray, np.ndarray, list]], cfg: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Static plate: per side, the median line over frames where that side was clearly measured.
        measured = [(box, scores, lines)] from measure_plate_box. Sides are matched across frames by their outward
        normal direction (detector corner order can change). A side never measured keeps the median rough side with
    support 0. Returns (box (4, 2), frames supporting each side)."""
    good = [m for m in measured if (m[1] > 0).sum() >= 2]
    if not good:
        return None
    ref_box = good[0][0]
    ref_c = ref_box.mean(0)
    ref = [_hesse(ref_box[k - 1], ref_box[k], ref_c)[0] for k in range(4)]  # side k: corner k-1 -> k
    th = [[] for _ in range(4)]
    rho = [[] for _ in range(4)]
    rough_th = [[] for _ in range(4)]  # unmeasured (hidden) sides keep the rough side, for completion later
    rough_rho = [[] for _ in range(4)]
    for box, sc, lines in good:
        c = box.mean(0)
        for (a, b), s in zip(lines, sc, strict=True):
            t, r = _hesse(a, b, c)
            k = int(np.argmin([abs(np.angle(np.exp(1j * (t - x)))) for x in ref]))
            (th if s > 0 else rough_th)[k].append(t)
            (rho if s > 0 else rough_rho)[k].append(r)
    support = np.array([len(x) for x in th])
    for k in range(4):
        if not th[k]:  # never measured in this window: rough side, support 0 -> complete_box rebuilds it
            th[k], rho[k] = rough_th[k], rough_rho[k]
    if any(len(x) == 0 for x in th):
        return None
    L = []
    for k in range(4):
        t = float(np.angle(np.median(np.cos(th[k])) + 1j * np.median(np.sin(th[k]))))
        n = np.array([np.cos(t), np.sin(t)])
        p = n * float(np.median(rho[k]))
        L.append((p, p + np.array([-n[1], n[0]])))
    box = np.array([_intersect(*L[k], *L[(k + 1) % 4]) for k in range(4)])
    return box, support


def _line_dist(L1: tuple[np.ndarray, np.ndarray], L2: tuple[np.ndarray, np.ndarray]) -> float:
    """Distance between two (near-parallel) lines, measured from the midpoint of L2's segment."""
    a, b = L1
    d = (b - a) / np.linalg.norm(b - a)
    m = (L2[0] + L2[1]) / 2 - a
    return float(abs(d[0] * m[1] - d[1] * m[0]))


def box_sides(box: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Side k runs from corner k-1 to corner k (the convention of fuse_boxes)."""
    return [(box[k - 1], box[k]) for k in range(4)]


def box_size(box: np.ndarray) -> tuple[float, float]:
    """(short, long) plate size in px: distances between opposite sides."""
    S = box_sides(box)
    return tuple(sorted((_line_dist(S[0], S[2]), _line_dist(S[1], S[3]))))


def complete_box(
    box: np.ndarray, support: np.ndarray, size: tuple[float, float] | None, ratio: float | None, cfg: dict
) -> tuple[np.ndarray, list[int]]:
    """Fix sides hidden for a whole window. A side is suspect when it was rarely measured (support < min_support;
    pipette/hand boundaries are already vetoed by colour in the edge search). A suspect side is rebuilt parallel to
    its opposite side when the pair's distance disagrees (> size_tol_px) with what this plate should measure: its
    own size in px (from its clean windows) or else the measured plate aspect ratio x the other, clearly measured
    dimension. Returns (box, rebuilt sides)."""
    ec = cfg["plate_edges"]
    S = box_sides(box)
    suspect = support < ec["min_support"]
    d = [_line_dist(S[0], S[2]), _line_dist(S[1], S[3])]
    lines = list(S)
    rebuilt = []
    for p_i, (i, j) in enumerate(((0, 2), (1, 3))):
        if not (suspect[i] ^ suspect[j]):
            continue  # both fine, or both suspect (nothing trustworthy to build from)
        other = (1, 3) if p_i == 0 else (0, 2)
        if size is not None:
            expected = size[1] if d[p_i] > d[1 - p_i] else size[0]  # plate aspect ~1.5: order survives a hidden side
        elif ratio is not None and not suspect[list(other)].any():
            od = d[1 - p_i]  # the other dimension, clearly measured
            expected = min((od * ratio, od / ratio), key=lambda e: abs(e - d[p_i]))
        else:
            continue
        if abs(d[p_i] - expected) <= ec["size_tol_px"]:
            continue
        keep, fix = (j, i) if suspect[i] else (i, j)
        a, b = S[keep]
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        if np.dot((S[fix][0] + S[fix][1]) / 2 - a, n) < 0:
            n = -n
        lines[fix] = (a + n * expected, b + n * expected)
        rebuilt.append(fix)
    # side k ends at corner k -> corner k = intersection of side k and side k+1
    return np.array([_intersect(*lines[k], *lines[(k + 1) % 4]) for k in range(4)]), rebuilt
