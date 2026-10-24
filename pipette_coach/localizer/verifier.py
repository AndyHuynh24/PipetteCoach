"""Learned grid verifier: is a candidate plate grid sitting exactly on the wells?

The plate is rectified with the candidate homography into a small top-down image in which a correct grid
always puts the 12x8 wells (and the plate rim) at the same pixels. A small CNN scores that image. Candidates
(the pipeline grid and its half-well / one-row / one-column shifts) are ranked by score. This targets the
failure the hand-made evidence cannot resolve at ~14 px per well: wells vs the gaps between them, and which
row/column is which when an outer row is hidden.
"""

from pathlib import Path

import cv2
import numpy as np

from pipette_coach.localizer.geometry import N_COLS, N_ROWS

# rectified view: plate outline (-1.6..12.6, -1.25..8.25 pitch) plus a margin, at PX_PER_PITCH
PX_PER_PITCH = 12
MARGIN = 0.4
X0, Y0 = -1.6 - MARGIN, -1.25 - MARGIN
OUT_W = int(round((N_COLS - 1 + 3.2 + 2 * MARGIN) * PX_PER_PITCH))
OUT_H = int(round((N_ROWS - 1 + 2.5 + 2 * MARGIN) * PX_PER_PITCH))
# candidate offsets (canonical pitch units): index 0 is the grid as given
OFFSETS = [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (1, 0), (-1, 0), (0, 1), (0, -1)]


def rectify(bgr: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Top-down view of the plate as the grid H sees it (OUT_H x OUT_W x 3)."""
    S = np.array([[PX_PER_PITCH, 0, -X0 * PX_PER_PITCH], [0, PX_PER_PITCH, -Y0 * PX_PER_PITCH], [0, 0, 1]], np.float64)
    M = S @ np.linalg.inv(np.asarray(H, np.float64))
    return cv2.warpPerspective(bgr, M, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def offset_H(H: np.ndarray, dx: float, dy: float, rot_deg: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """H with the canonical grid shifted by (dx, dy) pitch and rotated/scaled about the plate centre."""
    c = np.array([(N_COLS - 1) / 2, (N_ROWS - 1) / 2])
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * scale
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = c - R @ c + [dx, dy]
    return np.asarray(H, np.float64) @ T


def build_model():
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def to_tensor(imgs: list[np.ndarray]):
    import torch

    x = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]).astype(np.float32) / 255.0
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32))


class GridVerifier:
    def __init__(self, weights: str | Path, device: str | None = None):
        import torch

        from pipette_coach.localizer.detector import _default_device

        self.device = device or _default_device()
        self.model = build_model()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval().to(self.device)

    def scores(self, bgr: np.ndarray, Hs: list[np.ndarray]) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = to_tensor([rectify(bgr, H) for H in Hs]).to(self.device)
            return torch.sigmoid(self.model(x)).cpu().numpy().ravel()

    def best(self, bgr: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Best of H and its half-well / one-row / one-column shifts: (H_best, score, margin to runner-up)."""
        cands = [offset_H(H, dx, dy) for dx, dy in OFFSETS]
        s = self.scores(bgr, cands)
        order = np.argsort(-s)
        return cands[order[0]], float(s[order[0]]), float(s[order[0]] - s[order[1]])
