"""Liquid-release net: which well is liquid going into right now? One pass over the whole plate gives a score per well.

Input: the plate warped to a fixed grid, S px per well (well (col, row) centred in cell [row*S, (row+1)*S) x
[col*S, ...)), now and DT frames earlier, stacked to 6 channels. A ResNet18 (ImageNet weights) cut at stride 32 with
S = 32 has exactly one output per well: (8, 12) logits. Trained on SPACE-marked dispenses (scripts/record_dispenses.py,
scripts/build_liquid_dataset.py, scripts/train_liquid_net.py).
"""

import cv2
import numpy as np
import torch
from torch import nn

S = 32  # px per well pitch in the warped plate (= the net's stride: one output per well)
DT = 8  # frames between the two input images (~0.27 s at 30 fps)
MEAN = np.array([0.485, 0.456, 0.406] * 2, np.float32)
STD = np.array([0.229, 0.224, 0.225] * 2, np.float32)


def warp_plate(frame: np.ndarray, H: np.ndarray, s: int = S) -> np.ndarray:
    """(8*s, 12*s, 3) BGR: the plate on the well grid H (canonical well coords (col, row) -> image px)."""
    A = np.array([[s, 0, s / 2], [0, s, s / 2], [0, 0, 1]], np.float64)
    return cv2.warpPerspective(frame, A @ np.linalg.inv(H), (12 * s, 8 * s), flags=cv2.INTER_LINEAR)


def press_well(k: int) -> tuple[int, int]:
    """Grid (col, row) of press k for a plate filled one press per well, starting at grid (11, 7) and going up each
    column (row 7 -> 0), then the next column down (col 11 -> 0). pipette8's fill order."""
    return 11 - k // 8, 7 - k % 8


class LiquidNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        w = r.conv1.weight.data
        r.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        r.conv1.weight.data = torch.cat([w, w], 1) / 2  # starts as the pretrained net on the mean of the two frames
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.head = nn.Conv2d(512, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 6, 8*S, 12*S) normalised -> (B, 8, 12) logits."""
        return self.head(self.body(x))[:, 0]


def to_input(now: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Two warped BGR plates (uint8) -> (6, h, w) float32, RGB, ImageNet-normalised."""
    x = np.concatenate([now[..., ::-1], before[..., ::-1]], -1).astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def flip(now: np.ndarray, before: np.ndarray, target: np.ndarray, h: bool, v: bool):
    """Flip both plates and the (8, 12) target together (h: columns reversed, v: rows reversed)."""
    for do, ax in ((h, 1), (v, 0)):
        if do:
            now, before, target = np.flip(now, ax), np.flip(before, ax), np.flip(target, ax)
    return np.ascontiguousarray(now), np.ascontiguousarray(before), np.ascontiguousarray(target)


def shift_hue(img: np.ndarray, dh: int) -> np.ndarray:
    """Rotate every colour's hue by dh (OpenCV units, 0-180): blue liquid becomes green / pink / ..."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + dh) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------- v2: yes / no for ONE well (the well comes from the tip model + grid live, from the fill order in training)

CROP = 3  # wells per side of the crop around the well (the well in the middle, half its neighbours around it)


def crop_well(plate: np.ndarray, col: int, row: int, n: int = CROP, s: int = S) -> np.ndarray:
    """(n*s, n*s, 3): the warped plate around well (col, row), padded at the plate edge (edge pixels repeated)."""
    h = n // 2
    pad = np.pad(plate, ((h * s, h * s), (h * s, h * s), (0, 0)), mode="edge")
    return pad[row * s : (row + n) * s, col * s : (col + n) * s]


class LiquidCropNet(nn.Module):
    """Is liquid going into the middle well of the crop right now? (B, 6, CROP*S, CROP*S) -> (B,) logits."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        w = r.conv1.weight.data
        r.conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        r.conv1.weight.data = torch.cat([w, w], 1) / 2
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x).mean((2, 3)))[:, 0]
