"""Per-frame plate localization: detect once, then track; hold the last position through occlusion.

Live (fixed camera): PlateLocalizer.update(frame) every frame (causal).
Dataset clips (plate static within a clip): localize_clip_frames() fits the best frame, then tracks
forward and backward from it, so every frame gets a plate as long as one frame in the clip is good.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pipette_coach.localizer.edge_anchor import EdgeAnchor, shift_H
from pipette_coach.localizer.geometry import GRID, grid_error, is_plausible, orient, pitch_px, project, quad_to_H
from pipette_coach.localizer.grid_fit import fit_grid
from pipette_coach.localizer.openings import detect_rings, reconstruct_from_rings, refine_to_openings, ring_placement
from pipette_coach.localizer.refine import align_grid, snap_partial, snap_to_wells, track
from pipette_coach.localizer.rings import empty_fraction

_RANK = {"detector+align": 3, "detector+snap": 2, "grid_fit": 2, "detector+partial": 1}


def method_rank(method: str) -> int:
    """How much to trust a detection method (ignores the '+verifier' suffix)."""
    return _RANK.get(method.split("+")[0] + ("+" + method.split("+")[1] if "+" in method else ""), 0)


@dataclass
class PlateFit:
    """H maps canonical plate coords (protocol/wells.py) to image pixels; None when not found."""

    H: np.ndarray | None
    status: str  # "detected" | "tracked" | "held" | "not_found"
    method: str  # "detector+snap" | "detector+partial" | "grid_fit" | "track" | "memory" | reason if not found
    n_matched: int = 0  # wells supporting the fit
    det_conf: float = 0.0

    @property
    def found(self) -> bool:
        return self.H is not None


class PlateLocalizer:
    """SEARCHING -> (detect) -> TRACKING -> (too few wells visible) -> HELD -> ... -> SEARCHING."""

    def __init__(
        self,
        cfg: dict,
        detector=None,
        prior_H: np.ndarray | None = None,
        rows_toward: str | None = None,
        verifier=None,
    ):
        """prior_H: a plate fit remembered from earlier (e.g. a previous clip of the same plate). Only its
        orientation is used (which way the 12 columns run, which corner is A1), never its position, so a
        plate that moved in between is still re-detected."""
        self.cfg, self.detector, self.verifier = cfg, detector, verifier
        self.H: np.ndarray | None = None
        self.last_H: np.ndarray | None = prior_H  # keeps orientation / A1 labelling consistent
        self.rows_toward = rows_toward  # known image direction of rows A -> H (calibration / per-plate config)
        self.held = 0
        self.n_tracked = 0
        self.anchor = EdgeAnchor(cfg)  # plate-border reference: the grid only moves when the border moves
        self.anchored = False
        self.border_stale = False
        self.disagree = 0
        self.expected: np.ndarray | None = None  # grid the border predicted before a big-move re-detection

    def reset(self) -> None:
        """Forget the current position (keep the remembered orientation)."""
        self.H = None
        self.held = 0
        self.anchored = False
        self.border_stale = False
        self.disagree = 0
        self.expected = None

    def _plausible(self, H: np.ndarray, bgr: np.ndarray | None = None) -> bool:
        """Believable size, square lattice, and (if a frame is given) few predicted wells on empty bench."""
        dc = self.cfg["detector"]
        if not is_plausible(H, self.cfg["wells"]["pitch_px_range"], dc["max_axis_ratio"]):
            return False
        return bgr is None or empty_fraction(bgr, project(H, GRID), self.cfg) <= dc["max_empty_frac"]

    def detect(self, bgr: np.ndarray) -> PlateFit:
        fit = self._detect(bgr)
        if not fit.found:
            return fit
        confident = method_rank(fit.method) >= 3  # whole-grid alignment passed its own guards
        if self.verifier is not None:
            # every new detection is checked by the learned grid verifier (half-well / one-row / one-column
            # shifts); its correction is applied only when confident
            Hv, _, margin = self.verifier.best(bgr, fit.H)
            confident = margin >= self.cfg["verifier"]["min_margin"]
            if confident and grid_error(Hv, fit.H, ignore_labelling=False) > 0.25:
                return PlateFit(Hv, "detected", fit.method + "+verifier", fit.n_matched, fit.det_conf)
        if not confident and self.detector is not None:
            # placement not confident: detect the rings bottom-up from their edges (gaps between wells are
            # rejected: no edge all the way round at one radius), pick the placement the rings support, and
            # reconstruct the whole plate backward from the rings with a homography
            quad = self.detector.detect(bgr)
            if quad is not None:
                rings = detect_rings(bgr, quad[0], self.cfg)
                Hp, n_on, margin = ring_placement(rings, fit.H)
                if margin >= self.cfg["ring_edge"]["min_margin"]:
                    rec = reconstruct_from_rings(rings, Hp, self.cfg)
                    if rec is not None and self._plausible(rec[0], bgr):
                        tag = "+rings" if grid_error(rec[0], fit.H, ignore_labelling=False) > 0.25 else "+rings(kept)"
                        return PlateFit(rec[0], "detected", fit.method + tag, rec[1], fit.det_conf)
        return fit

    def _detect(self, bgr: np.ndarray) -> PlateFit:
        oc = self.cfg["orientation"]
        if self.detector is not None:
            det = self.detector.detect(bgr)
            if det is None:
                return PlateFit(None, "not_found", "no_detection")
            quad, conf = det
            H0 = quad_to_H(quad)
            # 1. whole-grid alignment on the radial-edge map: works for empty / filled / half-filled wells
            a = align_grid(bgr, H0, self.cfg, "detect")
            if a.ok and self._plausible(a.H, bgr):
                H = orient(a.H, oc["a1_rule"], oc["camera_mirrored"], self.last_H, self.rows_toward)
                H, _ = refine_to_openings(bgr, H, self.cfg)  # sub-pixel: centre on the opening edges
                return PlateFit(H, "detected", "detector+align", 96, conf)
            # 2. fallbacks for boxes far off (heavily covered plates): ring snap, then partial-plate rebuild
            s, method = snap_to_wells(bgr, H0, self.cfg), "detector+snap"
            if not (s.ok and self._plausible(s.H, bgr)):
                # box probably covers only the visible part of the plate: rebuild from visible wells + box edges
                s, method = snap_partial(bgr, H0, quad, self.cfg, self.last_H), "detector+partial"
            if not (s.ok and self._plausible(s.H, bgr)):
                return PlateFit(None, "not_found", f"rejected:{s.status}", s.n_matched, conf)
            H = orient(s.H, oc["a1_rule"], oc["camera_mirrored"], self.last_H, self.rows_toward)
            H, _ = refine_to_openings(bgr, H, self.cfg)  # centre on the well openings (circle edges)
            return PlateFit(H, "detected", method, s.n_matched, conf)
        g = fit_grid(bgr, self.cfg)
        if g is None or g.frac_on_rings < self.cfg["grid_fit"]["confident_frac"] or not self._plausible(g.H, bgr):
            return PlateFit(None, "not_found", "grid_fit")
        H = orient(g.H, oc["a1_rule"], oc["camera_mirrored"], self.last_H, self.rows_toward)
        H, _ = refine_to_openings(bgr, H, self.cfg)
        return PlateFit(H, "detected", "grid_fit", g.n_inliers)

    def _accept(self, fit: PlateFit) -> PlateFit:
        self.H = self.last_H = fit.H
        self.held = 0
        return fit

    def anchor_at(self, bgr: np.ndarray, H: np.ndarray) -> None:
        """Freeze the grid at H and remember the plate border in this frame."""
        self.H = self.last_H = np.asarray(H, np.float64)
        self.anchor.set(bgr, self.H)
        self.anchored, self.border_stale, self.disagree, self.held = True, False, 0, 0

    def _moved_px(self, H_new: np.ndarray) -> float:
        """Mean image displacement (px) of the 96 well centres between the current grid and H_new."""
        return float(np.linalg.norm(project(H_new, GRID) - project(self.H, GRID), axis=1).mean())

    def _anchored_update(self, bgr: np.ndarray) -> PlateFit | None:
        """Edge-anchored step. Returns None when full detection must run.

        The grid is kept exactly only while neither the plate border nor the visible wells moved by more than
        still_px. Any larger move is followed IMMEDIATELY (same frame) once a second cue agrees: the border
        shift is confirmed by the wells (ring tracking), or - when the border is hidden - the wells alone show
        it. Moves too large for tracking trigger detection at once."""
        ec = self.cfg["edge_anchor"]
        m = self.anchor.motion(bgr)
        if m.response < ec["min_response"]:  # border hidden (hand / pipette): ask the visible wells
            t = track(bgr, self.H, self.cfg)
            if t.ok and t.n_matched >= ec["wells_min_matches"] and self._moved_px(t.H) >= ec["wells_still_px"]:
                self.H = self.last_H = t.H  # plate pushed while its border is covered: follow now
                self.border_stale = True  # re-anchor the border once it is visible again
                return PlateFit(self.H, "tracked", "wells-moved (border hidden)", t.n_matched)
            self.held += 1
            if self.held > self.cfg["hold"]["max_frames"]:
                return None
            return PlateFit(self.H, "tracked", "edge-hidden (kept)", t.n_matched)
        self.held = 0
        if self.border_stale:  # border visible again after a wells-only move: new reference here
            self.anchor_at(bgr, self.H)
            return PlateFit(self.H, "tracked", "edge-reanchored", 0)
        shift = float(np.hypot(m.dx, m.dy))
        if shift < ec["still_px"]:
            self.disagree = 0
            return PlateFit(self.H, "tracked", "edge-still", 0)
        if shift > ec["redetect_pitch"] * pitch_px(self.H):
            self.expected = shift_H(self.H, m.dx, m.dy)  # where the border says the plate went
            return None  # big move: detect now
        H_new = shift_H(self.H, m.dx, m.dy)
        t = track(bgr, H_new, self.cfg)  # second, independent cue in the same frame: did the wells move too?
        if t.ok and grid_error(t.H, H_new, ignore_labelling=False) < ec["agree_pitch"]:
            self.disagree = 0
            self.anchor_at(bgr, t.H)
            return PlateFit(self.H, "tracked", "edge-moved", t.n_matched)
        wells_measurable = t.n_matched >= ec["wells_min_matches"]
        if not wells_measurable and m.response >= ec["border_alone_response"]:
            # wells hidden (e.g. multichannel pipette) but the border is matched very clearly: trust it now
            self.disagree = 0
            self.anchor_at(bgr, H_new)
            return PlateFit(self.H, "tracked", "edge-moved (border only)", t.n_matched)
        self.disagree += 1  # many visible wells say otherwise: probably a pipette/hand edge in the band
        if self.disagree >= ec["disagree_redetect"]:
            return None
        return PlateFit(self.H, "tracked", "edge-still (wells disagree)", 0)

    def update(self, bgr: np.ndarray) -> PlateFit:
        if self.H is not None and self.anchored:
            fit = self._anchored_update(bgr)
            if fit is not None:
                return fit
            self.anchored = False  # plate moved a lot / border hidden too long: fall back to detection
        n_vis = 0
        if self.H is not None:
            t = track(bgr, self.H, self.cfg)
            if not t.ok:  # ring tracker needs dark well bottoms; filled wells -> edge-based alignment instead
                a = align_grid(bgr, self.H, self.cfg, "track")
                if a.ok:
                    t = type(t)(a.H, t.n_matched, True, "aligned")
            if t.ok:
                H = t.H
                self.n_tracked += 1
                if self.n_tracked % self.cfg["openings"]["every_n_tracked"] == 0:
                    H, _ = refine_to_openings(bgr, H, self.cfg)  # re-centre on the openings periodically
                self.anchor_at(bgr, H)  # anchor was dropped: freeze again at the first good fit
                return PlateFit(H, "tracked", "track", t.n_matched)
            n_vis = t.n_matched
        fit = self.detect(bgr)
        if fit.found:
            ref = self.expected if self.expected is not None else self.H  # where the plate should be now
            self.expected = None
            if (
                ref is not None
                and grid_error(fit.H, ref, ignore_labelling=False) > self.cfg["edge_anchor"]["redetect_max_jump"]
            ):
                # we had a good grid; a re-detection far from where the plate should be is more likely the
                # half-well/row ambiguity than a real jump between consecutive frames: keep the expected grid
                self.anchor_at(bgr, ref)
                return PlateFit(self.H, "tracked", "kept previous (re-detection disagreed)", fit.n_matched)
            self.anchor_at(bgr, fit.H)
            return fit
        if self.H is not None and self.held < self.cfg["hold"]["max_frames"]:
            self.held += 1  # wells hidden (hand / pipette): keep the last good position
            return PlateFit(self.H, "held", "memory", n_vis)
        self.H = None
        return fit


def localize_clip_frames(
    frames: Sequence[np.ndarray],
    cfg: dict,
    detector=None,
    detect_every: int = 2,
    prior_H: np.ndarray | None = None,
    rows_toward: str | None = None,
    verifier=None,
) -> list[PlateFit]:
    """Plate fit for every frame of a dataset clip (offline: may use later frames to explain earlier ones).

    Detect on every `detect_every`-th frame, keep the best-supported detection, then track forward and
    backward from it with hold-through-occlusion. The plate moves < ~1 px within a clip.
    """
    loc = PlateLocalizer(cfg, detector, prior_H, rows_toward, verifier)
    best, best_i = None, -1
    for i in range(0, len(frames), detect_every):
        fit = loc.detect(frames[i])
        if fit.found and (
            best is None or (method_rank(fit.method), fit.n_matched) > (method_rank(best.method), best.n_matched)
        ):
            best, best_i = fit, i
    out = [PlateFit(None, "not_found", "no_good_frame") for _ in frames]
    if best is None:
        return out
    out[best_i] = best
    for order in (range(best_i + 1, len(frames)), range(best_i - 1, -1, -1)):
        loc.reset()
        loc.anchor_at(frames[best_i], best.H)  # grid frozen at the best frame; moves only if the border moves
        for i in order:
            out[i] = loc.update(frames[i])
    return out


def localize_clip(
    frames: Sequence[np.ndarray], cfg: dict, detector=None, prior_H: np.ndarray | None = None
) -> PlateFit:
    """One plate fit for a whole dataset clip: the best-supported detection."""
    fits = localize_clip_frames(frames, cfg, detector, detect_every=1, prior_H=prior_H)
    detected = [f for f in fits if f.status == "detected"]
    if not detected:
        return PlateFit(None, "not_found", "no_good_frame")
    return max(detected, key=lambda f: (method_rank(f.method), f.n_matched))
