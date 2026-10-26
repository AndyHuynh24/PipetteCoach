"""Live before/after liquid check with the trained model (scripts/train_well_compare.py): the well where the tip
stopped, before the tip came vs after it left.

Every frame with a grid, the plate is warped flat (32 px per well) and kept for keep_s. When the dwell rule logs a
stop (watch), the BEFORE view is taken from the frames before the tip arrived; once the tip has left (and stayed
away), the AFTER view, and the model says liquid or not. Two ways to take a view (mode):
  frame   one frame: `frames` frames before the tip arrived / after it left
  median  pixel median of the frames in before_s / after_s (seconds before arriving / after leaving): a tip, hand
          or glare that passes over the well for part of that time drops out
"""

from collections import deque
from contextlib import nullcontext

import numpy as np
import torch

from pipette_coach.liquid_net import LiquidCropNet, crop_well, to_input, warp_plate


class CompareCheck:
    def __init__(self, weights, mode: str = "median", device="cpu", thr: float = 0.5, frames: int = 3,
                 before_s=(1.0, 0.2), after_s=(0.2, 0.8), fps: float = 30.0, keep_s: float = 3.0, lock=None,
                 model=None):  # fmt: skip
        if model is None:
            model = LiquidCropNet(pretrained=False)
            model.load_state_dict(torch.load(weights, map_location="cpu")["state_dict"])
        self.model, self.device, self.mode, self.thr = model.to(device).eval(), device, mode, thr
        self.frames, self.before_s, self.after_s, self.fps, self.keep_s = frames, before_s, after_s, fps, keep_s
        self.lock = lock or nullcontext()
        self.hist: deque = deque()  # (t, warped plate)
        self.pending: dict = {}

    def _view(self, well, t_from: float, t_to: float, nearest: float | None = None):
        """Crop around well: the frame nearest `nearest`, or the pixel median of frames in [t_from, t_to]."""
        if nearest is not None:
            got = min(self.hist, key=lambda h: abs(h[0] - nearest), default=None)
            return None if got is None or abs(got[0] - nearest) > 0.2 else crop_well(got[1], *well)
        crops = [crop_well(p, *well) for t, p in self.hist if t_from <= t <= t_to]
        return np.median(np.stack(crops), 0).astype(np.uint8) if len(crops) >= 3 else None

    def watch(self, idx: int, well, t0: float) -> None:
        """A stop at `well` that began at t0 (the tip's arrival): take the before view now."""
        well = tuple(well)
        if self.mode == "frame":
            before = self._view(well, 0, 0, nearest=t0 - self.frames / self.fps)
        else:
            before = self._view(well, t0 - self.before_s[0], t0 - self.before_s[1])
        self.pending[idx] = {"well": well, "t1": t0, "before": before}

    def prob(self, after: np.ndarray, before: np.ndarray) -> float:
        x = torch.from_numpy(to_input(after, before)[None]).to(self.device)
        with self.lock, torch.no_grad():
            return float(torch.sigmoid(self.model(x)).cpu()[0])

    def update(self, t: float, frame: np.ndarray, H, tip_well) -> list:
        """One frame. Returns [(idx, {"filled": True / False / None, "p": ..., "after_s": ...})] decided now."""
        if H is not None:
            self.hist.append((t, warp_plate(frame, H)))
        while self.hist and t - self.hist[0][0] > self.keep_s:
            self.hist.popleft()
        out = []
        wait = self.frames / self.fps if self.mode == "frame" else self.after_s[1]
        for idx, p in list(self.pending.items()):
            if tip_well is not None and tuple(tip_well) == p["well"]:
                p["t1"] = t  # still there (or back)
                continue
            if t < p["t1"] + wait:
                continue
            if self.mode == "frame":
                after = self._view(p["well"], 0, 0, nearest=p["t1"] + self.frames / self.fps)
            else:
                after = self._view(p["well"], p["t1"] + self.after_s[0], p["t1"] + self.after_s[1])
            r = {"filled": None, "p": None, "after_s": round(t - p["t1"], 2)}
            if p["before"] is not None and after is not None:
                r["p"] = self.prob(after, p["before"])
                r["filled"] = r["p"] > self.thr
            out.append((idx, r))
            del self.pending[idx]
        return out
