"""Synthetic liquid in plate wells, for robustness tests and training augmentation (dataset wells are empty)."""

import cv2
import numpy as np

from pipette_coach.localizer.geometry import GRID, pitch_px, project

DYES = [(210, 70, 40), (40, 40, 210), (30, 200, 230), (170, 50, 160), (60, 180, 60), (20, 120, 240)]


def paint_fill(img: np.ndarray, H: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """SYNTHETIC liquid in the wells (same renderer as the fill robustness test). Paints each well in a
    small local patch (a full-frame mask per well made this ~100x slower)."""
    out = img.astype(np.float32)
    p = pitch_px(H)
    h, w = img.shape[:2]
    half = int(np.ceil(0.6 * p)) + 2
    for x, y in project(H, GRID):
        k = kind if kind != "mixed" else ["empty", "dye", "dark", "milky", "half", "half"][rng.integers(6)]
        if k == "empty":
            continue
        if k == "half":
            col, a, r = DYES[rng.integers(len(DYES))], 0.55, 0.24
        else:
            col, a, r = {"dye": (DYES[rng.integers(len(DYES))], 0.6, 0.33), "dark": ((40, 25, 20), 0.9, 0.33)}.get(
                k, ((235, 235, 235), 0.85, 0.33)
            )
        x0, y0 = int(x) - half, int(y) - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half >= w or y0 + 2 * half >= h:
            continue
        cx, cy = x - x0, y - y0
        m = np.zeros((2 * half, 2 * half), np.float32)
        cv2.circle(m, (int(round(cx * 16)), int(round(cy * 16))), int(r * p * 16), 1.0, -1, cv2.LINE_AA, 4)
        m = cv2.GaussianBlur(m, (0, 0), 0.8)[..., None] * a
        patch = out[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
        patch[:] = patch * (1 - m) + np.array(col, np.float32) * m
        if k == "half":
            c16 = (int(round(cx * 16)), int(round(cy * 16)))
            cv2.circle(patch, c16, int(r * p * 16), (60, 60, 60), 1, cv2.LINE_AA, 4)
    return np.clip(out, 0, 255).astype(np.uint8)
