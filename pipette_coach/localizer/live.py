"""Real-time plate + 12 x 8 grid tracker: one frame at a time, as from a live camera.

Per frame (cheap path, ~1 well-net pass):
  wells near the current grid (heatmap well net, plate crop at the locked scale) -> each visible well is
  identified on the remembered grid (memory gives its row / column, so which side of the plate is visible is
  known however little is visible) -> grid refitted to them (wells_first.track_grid). Hidden plate (arm over it):
  the last grid is held for up to hold_frames, then dropped.
Every det_every frames: the plate detector at the locked scale (full multi-scale search only when lost) - plate
  present? moving? A moving plate (box jumps, e.g. lifted / carried) shows NO grid until it is still again
  (blurry, tilted wells give wrong grids; nobody pipettes into a plate in mid-air).
Rebuild (expensive, rare): at start, after tracking is lost or the plate moved, and every verify_every frames as
  a check that may replace the tracked grid only if it shows the full 12 x 8 and fits clearly more wells. The
  first lock needs a view that settles length vs width (a full 12-well row, or a complete 8-well column with the
  plate edge at both ends); until then no grid is shown (tried every 3rd frame).
Anchor (memory of a still plate, anchor=True): the last grid that fitted most of the plate (>= anchor_wells wells).
  A grid > anchor_tol spacings away from it - a partial view (hand, sleeve) read as other rows / columns: on
  pipette1 the grid slid 1 column, or ~7 wells onto the paper, for up to ~2 s - is not taken: the wells are fitted
  from the anchor instead (or the grid is held). After a loss the plate is re-found at the anchor first (also after a
  "move": a sleeve sweeping over the plate looks like one). Only a near-full view (>= full_wells wells) of a
  different grid replaces the anchor (a plate really moved: the grid comes back once it is in full view).
Lock checks (lock_checks=True): every new grid from a rebuild is compared with itself moved by one column / row
  each way, on the wells found around it (+-1.5 spacings): the placement with clearly more wells on its nodes wins
  (a full view read one column off fits ~88 wells, the right one ~96). The first lock (no anchor yet) also needs
  >= anchor_wells wells on the grid: a plate half under a hand is not locked onto (virtual-camera test,
  scripts/eval_plate_cameras.py: partial views were read as other rows / columns).
No detector box (a plate / bench / camera the detector was not trained on, e.g. a clear plate on a blue mat): the
  box is taken from the wells instead - the well net over the whole frame at a few sizes (the plate filling 15-90%
  of the frame width, so any resolution), the biggest cluster of wells one spacing apart (wells_cluster_box).
Box check (box_tol): the plate detector's box is coarse (loose by 0-1 well) but rarely wrong; the grid comes from
  the wells, which can be read off the plate (pipette4: the tube rack's holes next to it, partial views fitted with
  a wrong tilt). A grid whose outer wells stick out of the box by > box_tol spacings is not taken (tracking, anchor,
  rebuild) and not held. A pushed plate (box shifted, same size) moves the anchor with the box.
Steady output (steady=True): the returned grid follows the tracked one smoothly (jitter averaged, a push followed at
  once); tracking itself always uses the raw fit.
Removed from the per-frame path (vs the offline render): multi-scale detector search, plate-edge snapping,
  repeated well-net passes; the outline comes from the grid and the plate's measured size.
"""

import time

import cv2
import numpy as np

from pipette_coach.localizer.detector import PlateDetector, boxes_of
from pipette_coach.localizer.geometry import GRID, grid_error, pitch_px, project
from pipette_coach.localizer.plate_edges import measure_plate_box
from pipette_coach.localizer.plate_grid import SEARCH_SCALES, detect_wells_net
from pipette_coach.localizer.wells_box import snap_frame
from pipette_coach.localizer.wells_first import find_plate, plate_outline, rebuild_from_visible, track_grid


def box_iou(a, b) -> float:
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def wells_angle(P: np.ndarray, pitch: float) -> float | None:
    """Angle (deg, -45..45) by which the frame must be rotated (cv2.getRotationMatrix2D convention) so the wells'
    rows and columns line up with the image axes: directions between wells one spacing apart (0.85-1.2 pitch:
    not the opening / floor ring pairs half a well apart diagonally, not diagonal neighbours), averaged mod 90 deg
    (circular mean of 4x the angle). None with < 8 such pairs."""
    P = np.asarray(P, np.float64)
    if len(P) < 4:
        return None
    v = (P[None] - P[:, None]).reshape(-1, 2)
    n = np.linalg.norm(v, axis=1)
    v = v[(n > 0.85 * pitch) & (n < 1.2 * pitch)]
    if len(v) < 16:  # each pair counted twice
        return None
    a = 4 * np.arctan2(v[:, 1], v[:, 0])
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 4)


def grid_angle(H: np.ndarray) -> float:
    """Angle (deg, -45..45) of the grid's rows / columns from the image axes."""
    d = project(H, GRID[[11]])[0] - project(H, GRID[[0]])[0]
    return (float(np.degrees(np.arctan2(d[1], d[0]))) + 45) % 90 - 45


PLATE_SPACINGS = 14.2  # a 96-well plate is ~14.2 well spacings wide (127.8 mm / 9 mm)
WELLS_BOX_CONF = 0.5  # confidence given to a box from the wells (no detector score)


def wells_cluster_box(P: np.ndarray, pitch: float, min_wells: int = 40, link: float = 1.5):
    """Box (x0, y0, x1, y1) around the biggest cluster of wells P that sit about one spacing apart, plus 0.9 spacing
    (to the plate's outer edge), or None when that cluster has < min_wells wells or the spacing is far from pitch
    (P from the well net at a size where a spacing should be ~pitch px). Scattered false wells (texture, a rack)
    are left out: a well joins the cluster only within link spacings of another."""
    P = np.asarray(P, np.float64).reshape(-1, 2)
    if len(P) < min_wells:
        return None
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(d, np.inf)
    p = float(np.median(d.min(1)))
    if not 0.6 * pitch <= p <= 1.6 * pitch:
        return None
    near = d < link * p
    label = np.full(len(P), -1)
    for i in range(len(P)):  # flood fill over "one spacing apart"
        if label[i] >= 0:
            continue
        label[i], todo = i, [i]
        while todo:
            for j in np.flatnonzero(near[todo.pop()] & (label < 0)):
                label[j] = i
                todo.append(j)
    big = np.bincount(label).argmax()
    C = P[label == big]
    if len(C) < min_wells:
        return None
    return np.array([*(C.min(0) - 0.9 * p), *(C.max(0) + 0.9 * p)])


class PlateTracker:
    def __init__(
        self,
        cfg: dict,
        det: PlateDetector,
        net,
        rows_toward: str | None = None,
        det_every: int = 6,
        verify_every: int = 60,
        hold_frames: int = 60,
        still_iou: float = 0.85,
        replace_margin: int = 5,
        track_every: int = 1,
        anchor: bool = True,
        anchor_tol: float = 0.5,
        anchor_wells: int = 60,
        full_wells: int = 85,
        lock_checks: bool = True,
        box_tol: float | None = 0.5,
        steady: bool = True,
        edge_every: int = 12,
    ):
        self.cfg, self.det, self.net, self.rows_toward = cfg, det, net, rows_toward
        self.det_every, self.verify_every, self.hold_frames = det_every, verify_every, hold_frames
        self.still_iou, self.replace_margin = still_iou, replace_margin
        self.track_every = track_every  # track the plate every n-th frame (1 = every frame)
        self.search_every = 3  # while no grid is locked, try to lock every 3rd frame (keeps the frame rate)
        # bad-grid test: >= bad_min_dets wells detected but < bad_share of them on the grid, bad_frames in a row
        self.bad_min_dets, self.bad_share, self.bad_frames = 20, 0.5, 5
        self.gate = cfg["plate_edges"]["well_gate_pitch"]
        self.use_anchor, self.anchor_tol = anchor, anchor_tol
        self.anchor_wells, self.full_wells = anchor_wells, full_wells
        self.lock_checks = lock_checks
        # plates at an angle: a failed rebuild is retried in a straightened frame when the wells lie > 5 deg off the
        # image axes (the plain rebuild copes up to ~10 deg); the well net gets a straightened region beyond 10 deg
        self.straighten_deg, self.wells_straighten_deg = 5.0, 10.0
        # box check: a grid whose outer wells stick out of the detector box by > box_tol spacings is off the plate
        # (the box is loose, 0-1 well: right grids on pipette4 <= 0.1, a grid on the rack holes next to it 0.8-2.5)
        self.box_tol = box_tol
        # steady output: the shown grid follows the tracked one smoothly - small changes (jitter, ~0.06 well a frame)
        # are averaged, a change of >= steady_follow spacings (the plate pushed) is followed at once
        self.steady, self.steady_follow, self.steady_min = steady, 0.3, 0.25
        # plate edges (edge_every frames, 0 = off): the plate's outer edges, searched inside the detector box, say
        # which wells the grid's outer rows / columns are; a grid read whole wells off is moved (see _edge_shift)
        self.edge_every, self.edge_size_tol, self.edge_clean = edge_every, 0.8, 0.35
        self.reset()

    def reset(self) -> None:
        self.i = 0
        self.H = None  # current grid, canonical -> image
        self.scale = None  # resize so a well spacing is ~pitch_px_nominal px (well net input)
        self.det_scale = None  # detector search scale that last found the plate
        self.box = self.conf = None
        self.box_src = None  # "det" (plate detector) or "wells" (box around the wells: the detector missed it)
        self.wells_box_scale = None  # resize at which _wells_box last found the wells
        self.prev_box = None
        self.moving = False
        self.last_ok = -(10**9)
        self.bad_run = 0
        self.last_out = None
        self.anchor = None  # last grid that fitted most of the still plate
        self.anchor_box = None  # detector box when the anchor was set (a pushed plate moves the anchor with the box)
        self.H_show = None  # steadied grid (output)
        self.edge_pending = None  # whole-well shift the plate edges asked for at the last check

    # ---- pieces
    def _detect(self, frame: np.ndarray):
        """Plate box and confidence: the plate detector's, or when it finds nothing, the box around the wells
        (_wells_box). A plate the detector missed before is looked for by its wells first (one cheap pass)."""
        wells_first = self.box_src == "wells"
        if wells_first and (box := self._wells_box(frame)) is not None:
            return box, WELLS_BOX_CONF
        box, conf = self._detect_plate(frame)
        if box is not None:
            self.box_src = "det"
            return box, conf
        if not wells_first and (box := self._wells_box(frame)) is not None:
            self.box_src = "wells"
            return box, WELLS_BOX_CONF
        return None, None

    def _wells_box(self, frame: np.ndarray) -> np.ndarray | None:
        """Plate box from the wells (wells_cluster_box), any resolution: the well net needs a spacing of ~pitch px,
        so the frame is resized for a plate 90, 60, 40, 27 or 18% of the frame wide (the last size that worked is
        tried first; a locked grid gives the size)."""
        pitch = self.cfg["wells"]["pitch_px_nominal"]
        sizes = [pitch * PLATE_SPACINGS / (f * frame.shape[1]) for f in (0.9, 0.6, 0.4, 0.27, 0.18)]
        first = self.scale or self.wells_box_scale
        for s in ([first] if first else []) + sizes:
            fs = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            h, w = fs.shape[:2]
            P = detect_wells_net(fs, np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64), self.cfg, self.net)
            box = wells_cluster_box(P, pitch)
            if box is not None:
                self.wells_box_scale = s
                return np.clip(box / s, 0, [w / s, h / s, w / s, h / s])
        return None

    def _detect_plate(self, frame: np.ndarray):
        scales = [self.det_scale] if self.det_scale is not None else list(SEARCH_SCALES)
        for attempt in (scales, [s for s in SEARCH_SCALES if s != self.det_scale]):
            for s0 in attempt:
                g = frame if s0 == 1.0 else cv2.resize(frame, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA
                                                       if s0 < 1 else cv2.INTER_LINEAR)  # fmt: skip
                r = self.det.model.predict(g, imgsz=self.det.imgsz, conf=self.det.min_conf, verbose=False,
                                           device=self.det.device)[0]  # fmt: skip
                q, c = boxes_of(r)
                if len(c):
                    k = int(np.argmax(c))
                    qq = q[k] / s0
                    self.det_scale = s0
                    return np.array([*qq.min(0), *qq.max(0)]), float(c[k])
            if self.det_scale is None:
                break
        return None, None

    def _wells(self, frame: np.ndarray, H: np.ndarray, s: float) -> np.ndarray:
        return self._wells_in(frame, project(H, GRID[[0, 11, 95, 84]]), s)

    def _wells_in(self, frame: np.ndarray, quad: np.ndarray, s: float) -> np.ndarray:
        """Well centres (frame px) inside quad (frame px; 4 corners in grid order) at resize s. The well net only
        works on plates square to the image (a 30 deg plate: 14 wells found instead of 112), so a plate lying at
        an angle is straightened first (only its region is turned), and the wells turned back."""
        a = 0.0
        if self.lock_checks:
            d = quad[1] - quad[0]
            a = (float(np.degrees(np.arctan2(d[1], d[0]))) + 45) % 90 - 45  # as grid_angle
        if abs(a) <= self.wells_straighten_deg:
            fs = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            return detect_wells_net(fs, quad * s, self.cfg, self.net) / s
        # region around the plate (+2 spacings), turned about its centre so the wells line up with the image axes
        pad = 2.0 * np.linalg.norm(quad[1] - quad[0]) / 11
        x0, y0 = np.maximum(np.floor(quad.min(0) - pad), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(quad.max(0) + pad), frame.shape[1::-1]).astype(int)
        A = cv2.getRotationMatrix2D(((x1 - x0) / 2, (y1 - y0) / 2), a, s)  # turn + resize in one warp
        A[:, 2] += np.array([x1 - x0, y1 - y0]) * (s - 1) / 2  # keep the turned region centred after scaling
        w, h = int((x1 - x0) * s), int((y1 - y0) * s)
        crop = cv2.warpAffine(frame[y0:y1, x0:x1], A, (w, h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                              borderValue=(90, 90, 90))  # fmt: skip
        q = cv2.transform((quad - [x0, y0])[None], A)[0]
        P = detect_wells_net(crop, q, self.cfg, self.net)
        if not len(P):
            return P.reshape(0, 2)
        back = cv2.invertAffineTransform(A)
        return cv2.transform(P[None].astype(np.float64), back)[0] + [x0, y0]

    def _rebuild(self, frame: np.ndarray, box: np.ndarray, conf: float) -> dict | None:
        if self.lock_checks and self.H is not None and abs(grid_angle(self.H)) > self.wells_straighten_deg:
            # periodic check of a plate lying at an angle: its angle is known from the grid (no search)
            return self._rebuild_rotated(frame, box, grid_angle(self.H))
        quad = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
        r = find_plate([frame], self.det, self.net, self.cfg, boxes=[(self.det_scale or 1.0, quad, conf)],
                       snap_edges=False)  # fmt: skip
        if r is not None:
            out = rebuild_from_visible(r, box, self.cfg, self.rows_toward)
            # the plain rebuild is only trusted for a plate within ~10 deg of the image axes (it reasons with the
            # axis-aligned box); a grid further off goes the straightened way below
            if out is not None and (
                not self.lock_checks or abs(grid_angle(out["H_grid"])) <= self.wells_straighten_deg
            ):
                H = self._sharpen(frame, out["H_grid"], r["scale"])
                return {"H": H, "scale": r["scale"], "counts": out["counts"], "basis": out["basis"]}
        if self.lock_checks:
            # not rebuilt: maybe a plate lying at an angle. The lattice fit itself needs the plate within ~20 deg of
            # the image axes: when it finds few wells and the box is not plate-shaped (a plate within ~10 deg of
            # the axes has a box ~1.5:1), also look in the frame turned by +-30 deg (one of 0 / +30 / -30 is
            # always within 15 deg)
            th = 0.0
            wb, hb = box[2] - box[0], box[3] - box[1]
            if (r is None or len(r["ij"]) < 60) and max(wb, hb) / max(min(wb, hb), 1) < 1.35:
                for t in (30.0, -30.0):
                    rt = self._find_turned(frame, box, t)
                    if rt is not None and (r is None or len(rt["ij"]) > len(r["ij"])):
                        r, th = rt, t
            if r is None:
                return None
            a = wells_angle(r["wells"], self.cfg["wells"]["pitch_px_nominal"] * r["scale_fit"] / r["scale"])
            total = th + (a or 0.0)
            if abs(total) > self.straighten_deg:
                return self._rebuild_rotated(frame, box, total)
        return None

    def _find_turned(self, frame: np.ndarray, box: np.ndarray, angle: float) -> dict | None:
        """find_plate in the frame turned by angle (deg) about the plate box centre (turned-frame coordinates)."""
        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        fr = cv2.warpAffine(frame, cv2.getRotationMatrix2D(c, angle, 1.0), frame.shape[1::-1], borderValue=(90, 90, 90))
        det_scale = self.det_scale
        b, cf = self._detect(fr)
        self.det_scale = det_scale
        if b is None:
            return None
        quad = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        return find_plate([fr], self.det, self.net, self.cfg, boxes=[(det_scale or 1.0, quad, cf)], snap_edges=False)

    def _rebuild_rotated(self, frame: np.ndarray, box: np.ndarray, angle: float) -> dict | None:
        """Rebuild for a plate lying at an angle: rebuild_from_visible reasons with the detector's axis-aligned box
        (which end of a side is the plate's real end), which only fits a plate square to the image (tolerance
        ~10 deg). Rotate the frame about the plate so its wells line up with the image axes, detect + rebuild
        there, and map the grid back (A1 rule applied in the original image)."""
        from pipette_coach.localizer.geometry import orient

        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        R = np.eye(3)
        det_scale = self.det_scale
        for _ in range(2):  # a second turn if the wells still lie > straighten_deg off (rough first angle)
            R = np.vstack([cv2.getRotationMatrix2D(c, angle, 1.0), [0, 0, 1]]) @ R
            fr = cv2.warpAffine(frame, R[:2], frame.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(90, 90, 90))
            box_r, conf_r = self._detect(fr)
            self.det_scale = det_scale
            if box_r is None:
                return None
            quad = np.array([[box_r[0], box_r[1]], [box_r[2], box_r[1]], [box_r[2], box_r[3]], [box_r[0], box_r[3]]])
            r = find_plate([fr], self.det, self.net, self.cfg, boxes=[(det_scale or 1.0, quad, conf_r)],
                           snap_edges=False)  # fmt: skip
            pitch = None if r is None else self.cfg["wells"]["pitch_px_nominal"] * r["scale_fit"] / r["scale"]
            angle = None if r is None else wells_angle(r["wells"], pitch)
            if angle is None or abs(angle) <= self.straighten_deg:
                break
        out = None if r is None else rebuild_from_visible(r, box_r, self.cfg)
        if out is None:
            return None
        oc = self.cfg["orientation"]
        H = orient(np.linalg.inv(R) @ out["H_grid"], oc["a1_rule"], oc["camera_mirrored"], rows_toward=self.rows_toward)
        H = self._sharpen(frame, H, r["scale"])
        return {"H": H, "scale": r["scale"], "counts": out["counts"], "basis": out["basis"]}

    def _sharpen(self, frame: np.ndarray, H: np.ndarray, s: float) -> np.ndarray:
        """Snap the outline to the plate's real edges and refit ONE homography to the measured corners (both
        adjacent sides found, close to the wells' outline) plus every well on the grid. Perspective comes from
        the fit; scale and ratio from the plate's known size. Only a refinement (<= 0.3 spacing): on these clips
        the strongest straight edges near the plate are often the pipette body or the inner wall, so edges are
        not allowed to move the grid by whole wells or reject it (tried 2026-10-08: coverage fell from 95 % to
        35 %). Edges not found cleanly -> keep the wells' grid."""
        p = self.cfg["wells"]["pitch_px_nominal"]
        fs = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        seed = plate_outline(H, self.cfg) * s
        ecfg = {**self.cfg, "plate_edges": {**self.cfg["plate_edges"], "search_pitch": 0.6, "use_colour": False}}
        quad, sc, _ = measure_plate_box(fs, seed, p, ecfg)
        ok = (sc > 0) & (np.roll(sc, 1) > 0) & (np.linalg.norm(quad - seed, axis=1) < 0.6 * p)
        long_p, short_p = self.cfg["plate_edges"]["plate_size_pitch"]
        ru, rv = (long_p - 11) / 2, (short_p - 7) / 2
        rect = np.array([[-ru, -rv], [11 + ru, -rv], [11 + ru, 7 + rv], [-ru, 7 + rv]], np.float32)
        matched, pos = self._matched(H, self._wells(frame, H, s))
        src = np.concatenate([GRID[matched], rect[ok]]).astype(np.float32)
        dst = np.concatenate([pos[matched], quad[ok] / s]).astype(np.float32)
        if len(src) < 8 or ok.sum() < 2:
            return H
        H2, _ = cv2.findHomography(src, dst, cv2.RANSAC, 0.15 * pitch_px(H))
        if H2 is None or grid_error(H2, H, ignore_labelling=False) > 0.3:  # sanity: a refinement, not a jump
            return H
        return H2

    def _best_shift(self, frame: np.ndarray, H: np.ndarray, s: float) -> tuple[np.ndarray, int]:
        """The grid or its one-column / one-row shift that puts clearly more detected wells on its nodes (wells
        searched +-1.5 spacings around the grid, so a missing outer column is seen). Returns (H, wells on it)."""
        wide = project(H, np.array([[-1.5, -1.5], [12.5, -1.5], [12.5, 8.5], [-1.5, 8.5]]))
        dets = self._wells_in(frame, wide, s)
        best = (H, int(self._matched(H, dets)[0].sum()))
        for _ in range(2):
            moves = [best[0] @ np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float)
                     for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]  # fmt: skip
            n, k = max((int(self._matched(m, dets)[0].sum()), k) for k, m in enumerate(moves))
            if n < best[1] + 3:
                break
            best = (moves[k], n)
        return best

    def _near_anchor(self, H: np.ndarray) -> bool:
        return self.anchor is None or grid_error(H, self.anchor, ignore_labelling=False) <= self.anchor_tol

    def _from_anchor(self, tr, dets: np.ndarray):
        """Tracked grid (H, wells) checked against the anchor: kept if close to it, else refitted from the anchor
        (None if that does not work either: the caller holds the last grid). First, a plate pushed since the anchor
        was set (detector box shifted, same size): the grid is refitted from the anchor moved with the box. A push
        of ~1 well cannot be told from the wells alone (the lattice repeats: 1.2 wells read as 0.2), the box can."""
        if not self.use_anchor or self.anchor is None or tr is None:
            return tr
        moved = self._anchor_with_box()
        if moved is not None:
            alt = track_grid(moved, dets, self.cfg)
            if alt is not None and grid_error(alt[0], moved, ignore_labelling=False) <= self.anchor_tol:
                if self._box_ok(alt[0]):
                    self._set_anchor(moved)
                    return alt
        if self._near_anchor(tr[0]):
            return tr
        alt = track_grid(self.anchor, dets, self.cfg)
        return alt if alt is not None and self._near_anchor(alt[0]) else None

    def _set_anchor(self, H: np.ndarray) -> None:
        self.anchor, self.anchor_box = H, self.box

    def _anchor_with_box(self) -> np.ndarray | None:
        """The anchor moved by the detector box's shift since the anchor was set. None: no box, the box moved
        < 0.6 spacing, or it changed size by > 0.5 spacing (a hand hiding part of the plate, not a push)."""
        if self.box is None or self.anchor_box is None:
            return None
        p = pitch_px(self.anchor)
        size = (self.box[2:] - self.box[:2]) - (self.anchor_box[2:] - self.anchor_box[:2])
        d = (self.box[:2] + self.box[2:] - self.anchor_box[:2] - self.anchor_box[2:]) / 2
        if np.linalg.norm(d) < 0.6 * p or np.abs(size).max() > 0.5 * p:  # the box itself wobbles ~0.4 spacing
            return None
        return np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1]]) @ self.anchor

    def _edge_shift(self, frame: np.ndarray, H: np.ndarray) -> tuple[int, int] | None:
        """Whole wells (columns, rows) by which the plate's outer edges say the grid is off (0 along an axis not
        measured cleanly), or None (neither axis). The edges are searched within 1.5 spacings of the grid's plate
        outline, in the frame cropped to the detector box (+0.3 spacing; nothing outside the box - rack, hands - can
        be taken as an edge), and mapped into grid coordinates. Per axis both opposite edges must be found, their
        distance must match the plate's size (+-edge_size_tol spacings), and their middle must sit a whole number of
        wells (+-edge_clean) from the middle of the 12 x 8 wells - opposite edges together are free of the rim-size
        and inner-wall uncertainty. On pipette4 right grids read 0 (+0.0-0.5 along the short side), a grid one row
        off 1.04-1.07."""
        p = pitch_px(H)
        nominal = self.cfg["wells"]["pitch_px_nominal"]
        x0, y0 = np.maximum(np.floor(self.box[:2] - 0.3 * p), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(self.box[2:] + 0.3 * p), frame.shape[1::-1]).astype(int)
        if x1 - x0 < 4 * p or y1 - y0 < 4 * p:
            return None
        pad = int(2 * p)
        crop = cv2.copyMakeBorder(frame[y0:y1, x0:x1], pad, pad, pad, pad, cv2.BORDER_REPLICATE)  # flat outside
        s = nominal / p
        fs = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        off = np.array([x0 - pad, y0 - pad], float)
        ec = {**self.cfg["plate_edges"], "search_pitch": 1.5, "step_px": 1.0, "n_samples": 30, "use_colour": False}
        _, sc, lines = measure_plate_box(
            fs, (plate_outline(H, self.cfg) - off) * s, nominal, {**self.cfg, "plate_edges": ec}
        )
        Hinv = np.linalg.inv(H)
        mid = [cv2.perspectiveTransform((((a + b) / 2 / s) + off)[None, None], Hinv)[0, 0] for a, b in lines]
        long_p, short_p = self.cfg["plate_edges"]["plate_size_pitch"]
        out = []
        # sides: 0 top (y = -rim), 1 right (x = 11 + rim), 2 bottom (y = 7 + rim), 3 left (x = -rim). Each axis on
        # its own: one not measured cleanly is left as it is (0)
        for lo, hi, ax, size, n in ((3, 1, 0, long_p, 12), (0, 2, 1, short_p, 8)):
            a, b = mid[lo][ax], mid[hi][ax]
            d = (a + b) / 2 - (n - 1) / 2
            clean = sc[lo] > 0 and sc[hi] > 0 and abs((b - a) - size) <= self.edge_size_tol
            out.append(int(round(d)) if clean and abs(d - round(d)) <= self.edge_clean else None)
        return None if out == [None, None] else (out[0] or 0, out[1] or 0)

    def _box_ok(self, H: np.ndarray) -> bool:
        """The grid's outer wells (their edges) lie inside the detector box, give or take box_tol spacings. The box
        is the plate as the detector sees it now; the grid is placed from the wells, which can be read off the plate
        (rack holes, a partial view fitted with a wrong tilt). No box (not detected) -> no check."""
        if self.box_tol is None or self.box is None or H is None:
            return True
        e = project(H, np.array([[-0.5, -0.5], [11.5, -0.5], [11.5, 7.5], [-0.5, 7.5]]))
        b, p = self.box, pitch_px(H)
        lo, hi = e.min(0), e.max(0)
        # an axis where the box is shorter than the grid's wells is cut by a hand (the detector boxes the VISIBLE
        # plate): no check along it
        full = (b[2:] - b[:2]) >= (hi - lo) - 0.5 * p
        out = np.maximum(b[:2] - lo, hi - b[2:])
        return bool((out[full] <= self.box_tol * p).all())

    def _steadied(self, H: np.ndarray | None) -> np.ndarray | None:
        """Shown grid: the corners move a share of the way to the tracked grid's - steady_min for jitter, all the way
        for a change of >= steady_follow spacings (a pushed plate is followed at once)."""
        if not self.steady or H is None or self.H_show is None:
            self.H_show = H
            return H
        idx = [0, 11, 95, 84]
        a, b = project(self.H_show, GRID[idx]), project(H, GRID[idx])
        d = float(np.linalg.norm(a - b, axis=1).mean()) / pitch_px(H)
        w = min(1.0, max(self.steady_min, d / self.steady_follow))
        self.H_show = cv2.getPerspectiveTransform(GRID[idx].astype(np.float32), (a + w * (b - a)).astype(np.float32))
        return self.H_show

    def _matched(self, H: np.ndarray, dets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, matched, pos = snap_frame(H, dets, self.gate, 10**9)
        return matched, pos

    # ---- main
    def update(self, frame: np.ndarray) -> dict:
        t0 = time.perf_counter()
        i = self.i
        self.i += 1
        if self.H is not None and self.track_every > 1 and i % self.track_every and self.last_out is not None:
            # plate on the table barely moves between frames: reuse the last grid (wells as last seen)
            return {**self.last_out, "ms": (time.perf_counter() - t0) * 1000}
        status = "no plate" if self.box is None else "waiting for a clear view"
        searching = self.H is None and not self.moving
        if (searching and i % self.search_every == 0) or (not searching and i % self.det_every == 0):
            self.box, self.conf = self._detect(frame)
            if self.box is not None:
                jumped = self.prev_box is not None and box_iou(self.box, self.prev_box) < self.still_iou
                # a hand changes the VISIBLE plate box without moving the plate: only a box jump whose wells no
                # longer sit on the grid counts as moving
                if jumped and self.H is not None:
                    d = self._wells(frame, self.H, self.scale)
                    jumped = track_grid(self.H, d, self.cfg) is None
                    if self.use_anchor:  # few wells visible = covered (sleeve, hand), not moved: moved = plenty of
                        jumped = jumped and len(d) >= self.bad_min_dets  # wells, but off the grid
                # no grid yet: nothing to protect - just try to lock (the lock has its own checks)
                self.moving = jumped and self.H is not None
                self.prev_box = self.box
        if self.moving:
            # the plate moved: re-lock once it is still again. The anchor is kept: a sleeve sweeping over the plate also
            # looks like a move (box jumps, wells off) - the plate is then re-found at the anchor; after a real move
            # only a near-full view (>= full_wells) replaces it
            self.H = self.H_show = None
            return {"status": "plate moving", "H": None, "box": self.box, "ms": (time.perf_counter() - t0) * 1000}
        dets, tracked = None, False
        if self.H is not None:
            dets = self._wells(frame, self.H, self.scale)
            tr = self._from_anchor(track_grid(self.H, dets, self.cfg), dets)
            if tr is not None and not self._box_ok(tr[0]):
                tr = None  # wells read off the plate (e.g. rack holes next to it): not taken
            if tr is not None:
                self.H, self.last_ok, tracked, status = tr[0], i, True, f"tracked ({tr[1]} wells)"
                if self.use_anchor and tr[1] >= self.anchor_wells:
                    self._set_anchor(tr[0])
            elif i - self.last_ok <= self.hold_frames and self._box_ok(self.H):
                status = "held (plate hidden)"
            else:
                self.H = None
            # hidden plate vs bad grid: hidden -> few wells detected; bad grid -> plenty of wells detected, but most
            # are off the grid. A bad grid is dropped after a few such frames and the plate is locked again.
            if self.H is not None and len(dets) >= self.bad_min_dets:
                on = self._matched(self.H, dets)[0].sum()
                self.bad_run = self.bad_run + 1 if on < self.bad_share * len(dets) else 0
                if self.bad_run >= self.bad_frames:
                    self.H, self.bad_run, tracked, status = None, 0, False, "grid did not fit the wells: re-locking"
        if self.edge_every and self.H is not None and self.box is not None and i % self.edge_every == 0:
            # the plate edges ask for a whole-well move twice in a row: the grid (and the anchor it came from) move
            sh = self._edge_shift(frame, self.H)
            if sh is not None and sh != (0, 0) and sh == self.edge_pending:
                T = np.array([[1, 0, sh[0]], [0, 1, sh[1]], [0, 0, 1]], float)
                if self._box_ok(self.H @ T):
                    if self.anchor is not None and self._near_anchor(self.H):
                        self._set_anchor(self.anchor @ T)
                    self.H, sh = self.H @ T, None
                    status = f"moved to the plate edges ({status})"
            self.edge_pending = sh
        if self.use_anchor and self.H is None and self.anchor is not None and i % self.search_every == 0:
            # plate lost for a moment (hand, sleeve) and not moved: find it again where it was
            dets = self._wells(frame, self.anchor, self.scale)
            tr = track_grid(self.anchor, dets, self.cfg)
            if tr is not None and self._near_anchor(tr[0]) and self._box_ok(tr[0]):
                self.H, self.last_ok, status = tr[0], i, f"re-found at the anchor ({tr[1]} wells)"
        need = (self.H is None and i % self.search_every == 0) or (i % self.verify_every == 0 and tracked)
        if need and self.box is not None:
            new = self._rebuild(frame, self.box, self.conf)
            if new is not None and self.lock_checks:
                new["H"], new["n"] = self._best_shift(frame, new["H"], new["scale"])
            if new is not None:
                if self.H is None:
                    # first lock (start, or after the plate moved): only from a view that settles length vs
                    # width - a full 12-well row visible. A partial view (e.g. multichannel tips lined up next to
                    # the wells) can be read the wrong way round, and tracking would then keep the error.
                    accept = new["basis"] in ("row12", "complete8")
                    if accept and self.lock_checks and self.anchor is None:
                        accept = new["n"] >= self.anchor_wells  # first lock: most of the plate in view
                    if accept and self.use_anchor and not self._near_anchor(new["H"]):
                        d_new = self._wells(frame, new["H"], new["scale"])
                        accept = self._matched(new["H"], d_new)[0].sum() >= self.full_wells
                else:  # periodic check: replace only a clearly worse tracked grid, with a full-plate view
                    full = sorted(new["counts"]) == [8, 12]
                    d_new = self._wells(frame, new["H"], new["scale"])
                    n_new = self._matched(new["H"], d_new)[0].sum()
                    accept = full and n_new >= self._matched(self.H, dets)[0].sum() + self.replace_margin
                    if accept and self.use_anchor and not self._near_anchor(new["H"]):
                        accept = n_new >= self.full_wells  # against memory: only a near-full view
                accept = accept and self._box_ok(new["H"])
                if accept:
                    self.H, self.scale, self.last_ok = new["H"], new["scale"], i
                    status = f"rebuilt {new['counts']}"
                    dets = None
                    if self.use_anchor:  # becomes the anchor if it fits most of the plate
                        n_new = self._matched(new["H"], self._wells(frame, new["H"], new["scale"]))[0].sum()
                        if n_new >= self.anchor_wells:
                            self._set_anchor(new["H"])
        out = {"status": status, "H": self._steadied(self.H), "box": self.box}
        if self.H is not None:
            if dets is None:
                dets = self._wells(frame, self.H, self.scale)
            out["outline"] = plate_outline(out["H"], self.cfg)
            out["wells"] = self._matched(self.H, dets)
        out["ms"] = (time.perf_counter() - t0) * 1000
        self.last_out = out if self.H is not None else None
        return out
