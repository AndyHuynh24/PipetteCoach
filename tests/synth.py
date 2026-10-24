"""Synthetic top-view plate images with a known homography (no dataset frames in tests)."""

import cv2
import numpy as np

from pipette_coach.localizer.geometry import GRID, OUTLINE, project


def plate_H(cx: float, cy: float, pitch: float = 14.5, angle_deg: float = 8.0) -> np.ndarray:
    """Similarity homography canonical -> image: plate centred at (cx, cy), rotated, A1 top-left-ish."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * pitch
    t = np.array([cx, cy]) - R @ np.array([5.5, 3.5])
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]], np.float64)


def render_plate(H: np.ndarray, size: tuple[int, int] = (640, 480), seed: int = 0) -> np.ndarray:
    """Grey bench, slightly lighter plate body, wells as bright rims with dark centres, blur + noise."""
    w, h = size
    img = np.full((h, w, 3), (92, 98, 104), np.uint8)
    cv2.fillConvexPoly(img, np.round(project(H, OUTLINE)).astype(np.int32), (150, 153, 155))
    p = float(np.linalg.norm(project(H, GRID[1:2]) - project(H, GRID[:1])))
    for x, y in project(H, GRID):
        c = (int(round(x)), int(round(y)))
        cv2.circle(img, c, int(round(0.45 * p)), (185, 188, 190), -1, cv2.LINE_AA)
        cv2.circle(img, c, int(round(0.24 * p)), (70, 72, 75), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    noise = np.random.default_rng(seed).normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)
