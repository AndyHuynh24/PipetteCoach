"""Recolour a dye in a video (e.g. red liquid -> blue / green / clear) to train the tip model on many liquid colours
from one labelled recording. Geometry is untouched, so the video's tip labels hold for every colour.

The dye is found per pixel by hue AND saturation (soft weights, so edges blend): pipette6's red liquid is hue
165-180 and 0-2 (OpenCV 0-180) at saturation 60-200; most skin is hue 4-12. The pinkest skin shares the liquid's hue,
so a Lab check follows: the liquid leans magenta (a-b angle -2..19 deg) or is vivid (chroma 45-60), skin is warmer and
duller (angle 19-32, chroma 24-39). Wood and the plate are not saturated enough to count.
The new colour keeps each pixel's brightness and saturation (scaled per colour), so the liquid keeps its shading and
the gradient towards the tip end.
"""

import cv2
import numpy as np

# target hue (OpenCV 0-180, None = no colour), saturation scale, brightness scale
COLORS = {
    "blue": (112, 1.0, 0.8),
    "green": (60, 0.9, 0.9),
    "yellow": (27, 1.0, 1.1),
    "purple": (140, 0.9, 0.85),
    "orange": (13, 1.0, 1.05),
    "clear": (None, 0.08, 1.08),
}


def dye_weight(
    bgr: np.ndarray,
    hue=(158, 182),
    hue_ramp: float = 2.0,
    sat=(60, 120),
    min_v: int = 40,
    lab_angle=(14.0, 22.0),
    lab_chroma=(38.0, 46.0),
) -> np.ndarray:
    """(h, w) float32 in [0, 1]: how much each pixel is the dye. hue = (lo, hi), may wrap past 180 (182 = hue 2):
    full weight inside, 0 beyond hue_ramp outside (short: skin starts at hue 4); saturation ramps sat[0] -> sat[1].
    Lab gate: full weight for an a-b angle <= lab_angle[0] (magenta-leaning) or chroma >= lab_chroma[1] (vivid),
    none for an angle >= lab_angle[1] with chroma <= lab_chroma[0] (skin)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = hue
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    d = np.abs((h - mid + 90) % 180 - 90)  # circular distance to the range centre
    w_h = np.clip(1 - (d - half) / hue_ramp, 0, 1)
    w_s = np.clip((s - sat[0]) / max(sat[1] - sat[0], 1), 0, 1)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    angle, chroma = np.degrees(np.arctan2(b, a)), np.hypot(a, b)
    w_lab = np.maximum(np.clip((lab_angle[1] - angle) / (lab_angle[1] - lab_angle[0]), 0, 1),
                       np.clip((chroma - lab_chroma[0]) / (lab_chroma[1] - lab_chroma[0]), 0, 1))  # fmt: skip
    w = w_h * w_s * w_lab * (v >= min_v)
    return cv2.GaussianBlur(w.astype(np.float32), (3, 3), 0)


def recolor(bgr: np.ndarray, weight: np.ndarray, color: str) -> np.ndarray:
    """The dye pixels (weight) turned into `color` (a COLORS key); everything else unchanged."""
    target, s_scale, v_scale = COLORS[color]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    if target is not None:
        hsv[..., 0] = target
    hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_scale, 0, 255)
    new = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    w = weight[..., None]
    return np.clip(bgr * (1 - w) + new * w, 0, 255).astype(np.uint8)
