"""Remembering the plate across a session of dataset clips (same plate, same bench, recorded in sequence).

Pass 1 finds the plate's orientation (which way the 12 columns run, A1 corner) by majority vote over
the clips' best detections, preferring fully snapped (clean) fits; pass 2 localizes each clip with that
orientation as a prior. Only orientation is shared between clips, never position.
"""

import cv2
import numpy as np

from pipette_coach.data import clip_path
from pipette_coach.localizer.localizer import PlateFit, PlateLocalizer, method_rank
from pipette_coach.localizer.refine import _col_axis_angle


def read_frames(clip: str, every: int = 1) -> list[np.ndarray]:
    cap, out, i = cv2.VideoCapture(str(clip_path(clip))), [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        if i % every == 0:
            out.append(f)
        i += 1


def best_detection(frames: list[np.ndarray], cfg: dict, detector) -> PlateFit | None:
    loc, best = PlateLocalizer(cfg, detector), None
    for f in frames:
        fit = loc.detect(f)
        if fit.found and (
            best is None or (method_rank(fit.method), fit.n_matched) > (method_rank(best.method), best.n_matched)
        ):
            best = fit
    return best


def session_prior(clips: list[str], cfg: dict, detector, every: int = 4) -> np.ndarray | None:
    """Orientation prior for a plate session: the best fit from the majority 12-column direction."""
    fits = [f for f in (best_detection(read_frames(c, every), cfg, detector) for c in clips) if f is not None]
    if not fits:
        return None
    ang = np.array([_col_axis_angle(f.H) for f in fits])
    weight = np.array([3.0 if f.method != "detector+partial" else 1.0 for f in fits])  # clean fits count more

    def support(a: float) -> float:
        d = np.abs(ang - a)
        return float(weight[np.minimum(d, 180 - d) < 30].sum())

    k = int(np.argmax([support(a) for a in ang]))
    d = np.abs(ang - ang[k])
    members = [f for f, dd in zip(fits, np.minimum(d, 180 - d), strict=True) if dd < 30]
    return max(members, key=lambda f: (f.method != "detector+partial", f.n_matched)).H
