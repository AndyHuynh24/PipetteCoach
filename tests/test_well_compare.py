"""pipette_coach/well_compare.py CompareCheck timing with a stand-in model (no weights, no video)."""

import numpy as np
from torch import nn

from pipette_coach.well_compare import CompareCheck

H = np.array([[30.0, 0, 40], [0, 30, 40], [0, 0, 1]])  # wells 30 px apart


class Seen(nn.Module):
    """Logit = mean(after) - mean(before) of the input (normalised channels 0-2 vs 3-5), records calls."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return (x[:, :3].mean((1, 2, 3)) - x[:, 3:].mean((1, 2, 3))) * 10


def frame(v):
    return np.full((360, 480, 3), v, np.uint8)


def run(mode, brighten_at=None):
    """Tip on well (3, 2) from 1.0 s to 2.0 s; the plate gets brighter at brighten_at (= liquid)."""
    chk = CompareCheck(None, mode, model=Seen())
    out = []
    for i in range(120):
        t = i / 30
        well = (3, 2) if 1.0 <= t < 2.0 else None
        if abs(t - 1.4) < 1e-6:
            chk.watch(0, (3, 2), 1.0)
        v = 150 if brighten_at is not None and t >= brighten_at else 100
        out += chk.update(t, frame(v), H, well)
    return out


def test_liquid_seen_both_modes():
    for mode in ("frame", "median"):
        ((idx, r),) = run(mode, brighten_at=1.5)
        assert idx == 0 and r["filled"] is True


def test_no_change_no_liquid_and_decided_after_the_tip_left():
    for mode, wait in (("frame", 0.1), ("median", 0.8)):
        ((_, r),) = run(mode)
        assert r["filled"] is False and abs(r["after_s"] - wait) < 0.05
