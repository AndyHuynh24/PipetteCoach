"""Heatmap keypoint net for well centres (small U-Net, CenterNet-style output).

Input: a BGR crop of the plate at native resolution. Output: a per-pixel "well centre here" probability map; wells
are local maxima above a threshold, refined to sub-pixel with a quadratic fit. Fully convolutional, so any crop
size works. Trained by scripts/train_well_net.py on ring-confirmed wells (gaps between wells are negatives,
hidden wells are ignored), with synthetic fills and scale augmentation.
"""

from pathlib import Path

import cv2
import numpy as np


def build_model(base: int = 16):
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )  # fmt: skip

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8]
            self.e1, self.e2, self.e3, self.e4 = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
            self.pool = nn.MaxPool2d(2)
            self.u3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), block(c[3], c[2])
            self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), block(c[2], c[1])
            self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), block(c[1], c[0])
            self.out = nn.Conv2d(c[0], 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)

    return UNet()


def to_tensor(imgs: list[np.ndarray]):
    import torch

    x = np.stack(imgs).astype(np.float32) / 255.0
    x = (x - 0.45) / 0.25
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def peaks(prob: np.ndarray, thr: float, nms: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima above thr, sub-pixel by quadratic fit. Returns (points (N, 2) x,y, scores)."""
    mx = cv2.dilate(prob, np.ones((nms, nms), np.uint8))
    ys, xs = np.nonzero((prob == mx) & (prob >= thr))
    pts = []
    for y, x in zip(ys, xs, strict=True):
        dx = dy = 0.0
        if 0 < x < prob.shape[1] - 1:
            a, b, c = prob[y, x - 1], prob[y, x], prob[y, x + 1]
            den = a - 2 * b + c
            dx = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
        if 0 < y < prob.shape[0] - 1:
            a, b, c = prob[y - 1, x], prob[y, x], prob[y + 1, x]
            den = a - 2 * b + c
            dy = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
        pts.append((x + float(np.clip(dx, -0.5, 0.5)), y + float(np.clip(dy, -0.5, 0.5))))
    return np.array(pts, np.float32).reshape(-1, 2), prob[ys, xs]


class WellNet:
    def __init__(self, weights: str | Path, device: str | None = None, thr: float = 0.5):
        import torch

        from pipette_coach.localizer.detector import _default_device

        self.device = device or _default_device()
        self.model = build_model()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval().to(self.device)
        self.thr = thr

    def heatmap(self, crop: np.ndarray) -> np.ndarray:
        import torch

        h, w = crop.shape[:2]
        H8, W8 = (h + 7) // 8 * 8, (w + 7) // 8 * 8  # U-Net needs multiples of 8
        pad = cv2.copyMakeBorder(crop, 0, H8 - h, 0, W8 - w, cv2.BORDER_REFLECT)
        with torch.no_grad():
            out = torch.sigmoid(self.model(to_tensor([pad]).to(self.device)))[0, 0].cpu().numpy()
        return out[:h, :w]

    def detect(self, bgr: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Well centres (full-image px) and scores inside roi = (x0, y0, x1, y1)."""
        hh, ww = bgr.shape[:2]
        x0, y0, x1, y1 = max(roi[0], 0), max(roi[1], 0), min(roi[2], ww), min(roi[3], hh)
        if x1 - x0 < 8 or y1 - y0 < 8:  # region (almost) outside the image
            return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        P, S = peaks(self.heatmap(bgr[y0:y1, x0:x1]), self.thr)
        return P + [x0, y0], S
