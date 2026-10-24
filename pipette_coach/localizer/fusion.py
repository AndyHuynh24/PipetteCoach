"""Temporal fusion of high-confidence well circles into one stable plate homography.

Per-frame fits wobble when much of the plate is hidden (plate 2, multichannel: 5-23 px). The plate does not
move within a dataset clip, so each well's position is the median of its confident circle detections over
the clip, and one homography is fit (RANSAC) on those medians. A motion check reports frames where the
observed circles no longer agree with the fused positions (plate moved).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from pipette_coach.localizer.geometry import GRID, pitch_px, project
from pipette_coach.localizer.rings import plate_circles


def well_observations(bgr: np.ndarray, quad: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Well-centre evidence inside the plate box: (ring, edge).

    ring: ring-filter peaks - precise on empty wells, and the gaps between wells never fire it.
    edge: strong radial-edge circles with no ring peak nearby - needed for filled wells (no dark centre), but on
          empty plates the bright gaps between wells pass too, so edge evidence never decides grid placement when
          ring evidence exists."""
    import copy

    from pipette_coach.localizer.rings import find_well_candidates, in_polygon, well_area_polygon

    fc = cfg["fusion"]
    area = well_area_polygon(quad, cfg)
    x0, y0 = np.floor(area.min(0)).astype(int) - 2
    x1, y1 = np.ceil(area.max(0)).astype(int) + 2
    rcfg = copy.deepcopy(cfg)
    rcfg["wells"]["method"] = "ring"
    R, _, _ = find_well_candidates(bgr, rcfg, (int(x0), int(y0), int(x1), int(y1)))
    R = R[in_polygon(R, area)] if len(R) else R
    E, S, _ = plate_circles(bgr, quad, cfg)
    E = E[S / 100.0 >= fc["hi_ratio"]] if len(E) else E
    if len(R) and len(E):
        E = E[np.linalg.norm(E[:, None] - R[None], axis=2).min(1) >= 0.4 * cfg["wells"]["pitch_px_nominal"]]
    return R, E


@dataclass
class FusedPlate:
    H: np.ndarray  # canonical -> image, fit on fused per-well positions
    n_wells: int  # wells with a trusted fused position
    rms: float  # px, inliers
    loo: float  # px, median leave-one-out prediction error = expected error of a reconstructed well
    seen: np.ndarray  # (n_frames, 96) bool: well confidently seen as a circle in that frame
    moved_frames: list[int]  # frames whose observations disagree with the fused positions
    support_margin: float  # (best - best grid >= half a well away) / best evidence count: low = ambiguous placement


def _assign(obs_frames: list[np.ndarray], H: np.ndarray, gate: float) -> tuple[list[list[np.ndarray]], int]:
    grid = project(H, GRID)
    per_well: list[list[np.ndarray]] = [[] for _ in range(len(GRID))]
    n = 0
    for P in obs_frames:
        if len(P) == 0:
            continue
        d = np.linalg.norm(grid[:, None] - P[None], axis=2)
        j = d.argmin(1)
        for k in np.nonzero(d[np.arange(len(GRID)), j] < gate)[0]:
            per_well[k].append(P[j[k]])
            n += 1
    return per_well, n


def fuse_clip(
    frames: Sequence[np.ndarray], quads: Sequence[np.ndarray | None], candidates: Sequence[np.ndarray], cfg: dict
) -> FusedPlate | None:
    """One stable homography for a clip with a static plate.

    candidates: plate fits seen in the clip (e.g. every detection/tracking H). Their +-1 row/column shifts are
    added, and the candidate whose wells collect the most evidence over the WHOLE clip decides which well each
    circle belongs to (a multichannel pipette hides different wells in different frames). Each well's position
    is then the median of its observations and one homography is fit on those medians (RANSAC).
    """
    fc = cfg["fusion"]
    empty = (np.zeros((0, 2)), np.zeros((0, 2)))
    both = [well_observations(f, q, cfg) if q is not None else empty for f, q in zip(frames, quads, strict=True)]
    ring_frames = [r for r, _ in both]
    edge_frames = [e for _, e in both]
    # placement is decided by ring evidence; edge evidence only if the plate shows almost no rings (filled)
    use_ring = sum(len(r) for r in ring_frames) >= fc["min_ring_obs_per_frame"] * len(frames)
    obs_frames = ring_frames if use_ring else [np.concatenate([r, e]) if len(e) else r for r, e in both]
    shifts = [
        np.array([[1, 0, dc], [0, 1, dr], [0, 0, 1]], np.float64)
        for dc, dr in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
    ]
    uniq: list[np.ndarray] = []
    for H in candidates:
        if all(
            np.median(np.linalg.norm(project(H, GRID) - project(U, GRID), axis=1)) > 0.15 * pitch_px(H) for U in uniq
        ):
            uniq.append(np.asarray(H, np.float64))
    scored = []
    for H in uniq:
        for T in shifts:
            Hs = H @ T
            _, n = _assign(obs_frames, Hs, fc["assign_gate"] * pitch_px(Hs))
            scored.append((n, Hs))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    H_assign = scored[0][1]
    second = next(
        (
            n
            for n, Hs in scored[1:]
            if np.median(np.linalg.norm(project(Hs, GRID) - project(H_assign, GRID), axis=1)) > 0.5 * pitch_px(Hs)
        ),
        0,
    )
    p = pitch_px(H_assign)
    obs, _ = _assign(obs_frames, H_assign, fc["assign_gate"] * p)
    if use_ring:  # wells never seen as a ring (e.g. filled ones) may still be seen as edge circles
        e_obs, _ = _assign(edge_frames, H_assign, fc["assign_gate"] * p)
        obs = [o if len(o) >= fc["min_obs"] else o + e for o, e in zip(obs, e_obs, strict=True)]
    seen = np.zeros((len(frames), len(GRID)), bool)
    grid = project(H_assign, GRID)
    for i, P in enumerate(obs_frames):
        if len(P):
            d = np.linalg.norm(grid[:, None] - P[None], axis=2)
            seen[i] = d.min(1) < fc["assign_gate"] * p
    trusted = [k for k in range(len(GRID)) if len(obs[k]) >= fc["min_obs"]]
    if len(trusted) < fc["min_wells"]:
        return None
    med = np.array([np.median(np.array(obs[k]), axis=0) for k in trusted], np.float32)
    can = GRID[trusted]
    H, inl = cv2.findHomography(can, med, cv2.RANSAC, fc["ransac_frac"] * p)
    if H is None:
        return None
    inl = inl.ravel().astype(bool)
    rms = float(np.sqrt(np.mean(np.linalg.norm(project(H, can[inl]) - med[inl], axis=1) ** 2)))
    idx = np.nonzero(inl)[0]
    loo = []
    for k in idx[:: max(1, len(idx) // 24)]:
        others = idx[idx != k]
        Hk, _ = cv2.findHomography(can[others], med[others], 0)
        if Hk is not None:
            loo.append(float(np.linalg.norm(project(Hk, can[k : k + 1])[0] - med[k])))
    fused = project(H, GRID)
    moved = []
    for i, P in enumerate(obs_frames):
        if seen[i].sum() >= 8:
            d = np.linalg.norm(fused[seen[i]][:, None] - P[None], axis=2).min(1)
            if np.median(d) > fc["motion_gate"] * p:
                moved.append(i)
    support_margin = (scored[0][0] - second) / max(scored[0][0], 1)
    return FusedPlate(
        H, int(inl.sum()), rms, float(np.median(loo)) if loo else float("nan"), seen, moved, support_margin
    )
