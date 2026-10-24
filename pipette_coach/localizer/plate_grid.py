"""Plate box + 12 x 8 well grid for a clip (the method approved 2026-10-07; see docs/DECISIONS.md).

Per static window of a clip: the plate's own edges (plate_edges) -> heatmap-net well detections pooled over the
window -> lattice -> box completed from the wells (wells_box) -> 12 x 8 grid centred in the box, oriented (A1).
Used by the renders, the grid cache (scripts/cache_plate_grids.py) and the pipette-tip stage.
"""

import cv2
import numpy as np

from pipette_coach.localizer.detector import PlateDetector
from pipette_coach.localizer.geometry import orient
from pipette_coach.localizer.plate_edges import _line_dist, box_sides, fuse_boxes, measure_plate_box
from pipette_coach.localizer.rings import in_polygon
from pipette_coach.localizer.wells_box import box_from_wells, fit_lattice, grid_from_lattice, pool_points, snap_frame


def side_deviation(lines: list, scores: np.ndarray, box: np.ndarray) -> list[float]:
    """For each clearly measured side of a frame: distance of its midpoint to the nearest side line of box."""
    out = []
    for (a, b), s in zip(lines, scores, strict=True):
        if s > 0:
            out.append(min(_line_dist(S, (a, b)) for S in box_sides(box)))
    return out


def measure_clip(frames: list, det: PlateDetector, cfg: dict) -> list[dict]:
    """Windows of a static plate: [{start, end, box, support (share of frames per side)}]."""
    ec = cfg["plate_edges"]
    n = ec["every_n"]
    pitch = cfg["wells"]["pitch_px_nominal"]
    windows: list[dict] = []
    cur: list = []
    cur_idx: list[int] = []
    pending = 0
    for i in range(0, len(frames), n):
        d = det.detect(frames[i])
        if d is None:
            continue
        m = measure_plate_box(frames[i], d[0], pitch, cfg)
        if len(cur) >= 3:
            fb = fuse_boxes(cur, cfg)
            if fb is not None:
                dev = side_deviation(m[2], m[1], fb[0])
                moved = sum(x > ec["move_px"] for x in dev) >= 2
                pending = pending + 1 if moved else 0
                if pending >= ec["move_confirm"]:  # sampled frames in a row agree the plate moved: new window
                    k = ec["move_confirm"] - 1
                    windows.append({"idx": cur_idx[:-k], "meas": cur[:-k]})
                    cur, cur_idx, pending = cur[-k:], cur_idx[-k:], 0
        cur.append(m)
        cur_idx.append(i)
    if cur:
        windows.append({"idx": cur_idx, "meas": cur})
    # merge neighbouring windows whose median boxes agree: the split came from a hand/pipette, not a move
    merged: list[dict] = []
    for w in windows:
        if merged:
            fa, fb = fuse_boxes(merged[-1]["meas"], cfg), fuse_boxes(w["meas"], cfg)
            if fa is not None and fb is not None:
                lines_b = box_sides(fb[0])
                if max(side_deviation(lines_b, np.ones(4), fa[0])) <= ec["move_px"]:
                    merged[-1] = {"idx": merged[-1]["idx"] + w["idx"], "meas": merged[-1]["meas"] + w["meas"]}
                    continue
        merged.append(w)
    windows = merged
    out = []
    for w in windows:
        fb = fuse_boxes(w["meas"], cfg)
        if fb is None:
            continue
        box, sup = fb
        out.append({
            "start": w["idx"][0], "end": w["idx"][-1] + n, "box": box, "support": sup / len(w["meas"]),
        })  # fmt: skip
    return out


def grow(quad: np.ndarray, pitch: float, k: float) -> np.ndarray:
    """Quad grown outward by k pitch (along the centre -> corner directions)."""
    quad = np.asarray(quad, np.float64)
    c = quad.mean(0)
    return c + (quad - c) * (1 + k * pitch / np.linalg.norm(quad - c, axis=1, keepdims=True))


def detect_wells_net(frame: np.ndarray, quad: np.ndarray, cfg: dict, net) -> np.ndarray:
    """Heatmap-net well centres inside the (rough or measured) plate quad grown by half a pitch."""
    g = grow(quad, cfg["wells"]["pitch_px_nominal"], 0.5).astype(np.float32)
    x0, y0 = np.floor(g.min(0)).astype(int)
    x1, y1 = np.ceil(g.max(0)).astype(int)
    P, _ = net.detect(frame, (int(x0), int(y0), int(x1), int(y1)))
    return P[in_polygon(P, g)] if len(P) else P


def clip_grid_windows(
    frames: list, det: PlateDetector, net, cfg: dict, size_ls: tuple[float, float], rows_toward: str | None
) -> list[dict]:
    """[{start, end, box (4, 2), sides (L,T,R,B sources), H_grid (3x3, canonical -> image) or None}] per static
    window. size_ls = plate (long, short) in pitch units (measured; configs/localizer.yaml plate_edges)."""
    ec, oc = cfg["plate_edges"], cfg["orientation"]
    pitch = cfg["wells"]["pitch_px_nominal"]
    out = []
    for w in measure_clip(frames, det, cfg):
        pts = []
        for i in range(w["start"], min(w["end"], len(frames)), ec["every_n"]):
            d = det.detect(frames[i])
            if d is not None:
                pts.append(detect_wells_net(frames[i], d[0], cfg, net))
        wells = pool_points(pts, pitch, min_frames=2 if len(pts) >= 4 else 1)
        lat = fit_lattice(wells, tuple(cfg["wells"]["pitch_px_range"]))
        r = None if lat is None else box_from_wells(lat[0], lat[1], w["box"], w["support"], size_ls, (12, 8), cfg)
        H = None
        box, sides = w["box"], ["edge-only"] * 4
        if r is not None:
            box, sides = r[0], [r[1][n] for n in ("uL", "vT", "uR", "vB")]
            Hg = grid_from_lattice(lat[0], lat[1], r[2]["rect"], r[2]["u_long"], (12, 8))
            Hg = snap_frame(Hg, wells, ec["well_gate_pitch"])[0]
            H = orient(Hg, oc["a1_rule"], oc["camera_mirrored"], rows_toward=rows_toward)
        out.append({"start": w["start"], "end": w["end"], "box": np.asarray(box), "sides": sides, "H_grid": H})
    return out


def window_at(windows: list[dict], i: int) -> dict | None:
    return next((w for w in windows if w["start"] <= i < w["end"]), windows[-1] if windows else None)


SEARCH_SCALES = (1.0, 0.7, 0.5, 0.35, 0.25, 1.4, 2.0)  # try the trained camera distance first, then closer/farther


def plate_scale(frames: list, det: PlateDetector, cfg: dict) -> float | None:
    """Resize factor that brings the plate to the trained size (one well spacing ~ wells.pitch_px_nominal px).
    The plate detector only fires near the trained scale, so the frame is searched at several scales; the plate
    box (long side ~ plate_size_pitch[0] spacings) then gives the exact factor. None if no plate is found."""
    p_nom = cfg["wells"]["pitch_px_nominal"]
    long_pitch = cfg["plate_edges"]["plate_size_pitch"][0]
    for f in (frames[len(frames) // 2], frames[0], frames[-1]):
        for s0 in SEARCH_SCALES:
            g = f if s0 == 1.0 else cv2.resize(f, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA if s0 < 1
                                               else cv2.INTER_LINEAR)  # fmt: skip
            d = det.detect(g)
            if d is not None:
                q = d[0] / s0
                long_px = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]))
                return float(np.clip(p_nom / (long_px / long_pitch), 0.2, 3.0))
    return None


def clip_grid_windows_any_scale(
    frames: list,
    det: PlateDetector,
    net,
    cfg: dict,
    size_ls: tuple[float, float],
    rows_toward: str | None,
    scale: float | None = None,
) -> tuple[list[dict], float | None]:
    """clip_grid_windows for a camera at any distance: frames are rescaled to the trained plate size, the grid is
    fitted there and mapped back. Each window also keeps "H_scaled" (grid in the rescaled frames) and "scale"."""
    s = scale if scale is not None else plate_scale(frames, det, cfg)
    if s is None:
        return [], None
    if abs(s - 1) < 0.05:
        fs = frames
    else:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        fs = [cv2.resize(f, None, fx=s, fy=s, interpolation=interp) for f in frames]
    S_inv = np.diag([1 / s, 1 / s, 1.0])
    out = []
    for w in clip_grid_windows(fs, det, net, cfg, size_ls, rows_toward):
        H = None if w["H_grid"] is None else S_inv @ w["H_grid"]
        out.append({**w, "box": w["box"] / s, "H_grid": H, "H_scaled": w["H_grid"], "scale": s})
    return out, s
