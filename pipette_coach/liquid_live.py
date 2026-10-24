"""Live liquid check with the one-well model (pipette_coach/liquid_net.py LiquidCropNet): liquid going into the well
under the tip, decided WHILE it goes in (not after the tip leaves, like well_change.py).

Every frame with a grid, the plate is warped flat (32 px per well) and kept for ~1 s. When the tip is over a well,
the model compares that well now with ~0.27 s earlier (the input it was trained on). Decision: a well fires when it
has >= `need` frames above `thr` within the last `win_s` AND no neighbour has more (the tip point / grid can sit
half a well off, so a neighbour's crop sees the liquid too), and it has not fired in the last `rearm_s` (the tip
flickering between two wells does not count twice). Tuned on pipette8 val presses 64-79 (live run, 2026-10-24).
"""

from collections import deque
from contextlib import nullcontext

import numpy as np
import torch

from pipette_coach.liquid_net import DT, LiquidCropNet, crop_well, to_input, warp_plate


class ReleaseCheck:
    def __init__(self, weights, device="cpu", thr: float = 0.95, need: int = 4, win_s: float = 1.0,
                 rearm_s: float = 2.0, dt_s: float = DT / 30.0, keep_s: float = 1.0, lock=None,
                 model=None):  # fmt: skip
        if model is None:
            model = LiquidCropNet(pretrained=False)
            model.load_state_dict(torch.load(weights, map_location="cpu")["state_dict"])
        self.model = model.to(device).eval()
        self.device, self.thr, self.need, self.win_s, self.rearm_s = device, thr, need, win_s, rearm_s
        self.dt_s, self.keep_s = dt_s, keep_s
        self.lock = lock or nullcontext()
        self.hist: deque = deque()  # (t, warped plate)
        self.yes: deque = deque()  # (t, well) of frames above thr in the last win_s
        self.last_fire: dict = {}  # well -> time it last fired
        self.p = None  # last probability (for drawing)
        self.trace: list = []  # (t, col, row, p) per frame with a verdict: to tune the decision offline

    def _before(self, t: float):
        """The kept plate closest to t - dt_s (at least half of dt_s old), or None."""
        best = None
        for tt, plate in self.hist:
            if t - tt >= self.dt_s / 2 and (best is None or abs(t - tt - self.dt_s) < abs(t - best[0] - self.dt_s)):
                best = (tt, plate)
        return None if best is None else best[1]

    def prob(self, now: np.ndarray, before: np.ndarray, well) -> float:
        x = torch.from_numpy(to_input(crop_well(now, *well), crop_well(before, *well))[None]).to(self.device)
        with self.lock, torch.no_grad():
            return float(torch.sigmoid(self.model(x)).cpu()[0])

    def update(self, t: float, frame: np.ndarray, H, well):
        """One frame. Returns the well (col, row) when a dispense is decided in this frame, else None."""
        self.p = None
        if H is None:
            return None
        now = warp_plate(frame, H)
        self.hist.append((t, now))
        while self.hist and t - self.hist[0][0] > self.keep_s:
            self.hist.popleft()
        while self.yes and t - self.yes[0][0] > self.win_s:
            self.yes.popleft()
        if well is None:
            return None
        before = self._before(t)
        if before is None:
            return None
        self.p = self.prob(now, before, well)
        self.trace.append((round(t, 3), int(well[0]), int(well[1]), round(self.p, 4)))
        if self.p > self.thr:
            self.yes.append((t, well))
        count: dict = {}
        for _, w in self.yes:
            count[w] = count.get(w, 0) + 1
        if count.get(well, 0) < self.need or t - self.last_fire.get(well, -1e9) < self.rearm_s:
            return None
        c, r = well
        if any(count.get((c + dc, r + dr), 0) > count[well] for dc in (-1, 0, 1) for dr in (-1, 0, 1)):
            return None  # a neighbour has more evidence: wait (it may be the real well)
        self.last_fire[well] = t
        return well
