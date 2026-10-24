"""Edge anchor: keep the plate grid still unless the plate's own border moves.

Fixed camera + plate resting on the bench: the grid should only move when the plate moves. The reference is the
colour-edge image of a band around the plate outline at the moment the grid was confirmed. Each new frame, the
same band is phase-correlated with the reference: a confident ~zero shift keeps the grid exactly as it is (no
jitter), a confident real shift moves the grid by that motion, and a weak correlation (border hidden, e.g. a
hand over the plate) keeps the grid too. Hands and pipettes do not move the grid because only the border band
is compared, and the comparison is a single global shift (an occluder moving inside the band lowers the
correlation peak instead of shifting it).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pipette_coach.localizer.geometry import OUTLINE, pitch_px, plate_bbox, project


def edge_map(bgr: np.ndarray) -> np.ndarray:
    im = cv2.GaussianBlur(bgr, (0, 0), 1.0).astype(np.float32)
    mag = np.zeros(im.shape[:2], np.float32)
    for k in range(3):
        mag = np.maximum(
            mag, cv2.magnitude(cv2.Sobel(im[..., k], cv2.CV_32F, 1, 0), cv2.Sobel(im[..., k], cv2.CV_32F, 0, 1))
        )
    return mag


def band_mask(shape: tuple[int, int], H: np.ndarray, offset: tuple[int, int], band: float) -> np.ndarray:
    """1 inside a band of +-band pitch around the plate outline (in the ROI's coordinates)."""
    grow = np.array([[-band, -band], [band, -band], [band, band], [-band, band]], np.float32)
    outer = project(H, OUTLINE + grow) - offset
    inner = project(H, OUTLINE - grow) - offset
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(outer).astype(np.int32)], 1)
    cv2.fillPoly(m, [np.round(inner).astype(np.int32)], 0)
    return m.astype(np.float32)


@dataclass
class EdgeMotion:
    dx: float
    dy: float
    response: float  # normalised cross-correlation at the best shift: high = border clearly visible and matched


class EdgeAnchor:
    def __init__(self, cfg: dict):
        self.cfg = cfg["edge_anchor"]
        self.ref: np.ndarray | None = None
        self.roi: tuple[int, int, int, int] | None = None
        self.idx: tuple[np.ndarray, np.ndarray] | None = None
        self.R = 1

    def set(self, bgr: np.ndarray, H: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = plate_bbox(H, self.cfg["roi_margin"])
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        self.R = int(np.ceil(self.cfg["search_pitch"] * pitch_px(H)))
        self.roi = (x0, y0, x1, y1)
        mask = band_mask((y1 - y0, x1 - x0), H, (x0, y0), self.cfg["band"])
        R = self.R
        mask[:R], mask[-R:], mask[:, :R], mask[:, -R:] = 0, 0, 0, 0  # every shift must stay inside the ROI
        self.idx = np.nonzero(mask)
        e = edge_map(bgr[y0:y1, x0:x1])[self.idx]
        self.ref = (e - e.mean()) / (e.std() + 1e-6)

    def motion(self, bgr: np.ndarray) -> EdgeMotion | None:
        if self.ref is None:
            return None
        x0, y0, x1, y1 = self.roi
        cur = edge_map(bgr[y0:y1, x0:x1])
        ys, xs = self.idx
        R = self.R
        ncc = np.full((2 * R + 1, 2 * R + 1), -1.0)
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                c = cur[ys + dy, xs + dx]
                ncc[dy + R, dx + R] = float(np.dot(self.ref, c - c.mean()) / (c.std() + 1e-6) / len(c))
        j, i = np.unravel_index(np.argmax(ncc), ncc.shape)
        sub = []
        for a, b, cc in (
            (ncc[j, i - 1], ncc[j, i], ncc[j, i + 1]) if 0 < i < 2 * R else (0, 0, 0),
            (ncc[j - 1, i], ncc[j, i], ncc[j + 1, i]) if 0 < j < 2 * R else (0, 0, 0),
        ):
            den = a - 2 * b + cc
            sub.append(0.5 * (a - cc) / den if abs(den) > 1e-9 else 0.0)
        return EdgeMotion(
            i - R + float(np.clip(sub[0], -0.5, 0.5)), j - R + float(np.clip(sub[1], -0.5, 0.5)), float(ncc[j, i])
        )


def shift_H(H: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64) @ np.asarray(H, np.float64)


__all__ = ["EdgeAnchor", "EdgeMotion", "edge_map", "pitch_px", "shift_H"]
