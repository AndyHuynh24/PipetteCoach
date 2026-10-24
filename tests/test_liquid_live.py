"""pipette_coach/liquid_live.py ReleaseCheck decisions with a stand-in model (no weights, no video)."""

import numpy as np
import torch
from torch import nn

from pipette_coach.liquid_live import ReleaseCheck


class Fixed(nn.Module):
    """Answers the next logit from a list (one per call)."""

    def __init__(self, logits):
        super().__init__()
        self.logits = list(logits)

    def forward(self, x):
        return torch.tensor([self.logits.pop(0)], dtype=torch.float32)


H = np.array([[30.0, 0, 40], [0, 30, 40], [0, 0, 1]])  # wells 30 px apart
FRAME = np.zeros((360, 480, 3), np.uint8)


def run(logits, wells, fps=30.0):
    chk = ReleaseCheck(None, model=Fixed(logits))
    return [chk.update(i / fps, FRAME, H, w) for i, w in enumerate(wells)]


def test_fires_once_per_visit_after_enough_yes_frames():
    w = (3, 2)
    out = run([5] * 30, [w] * 30)  # first frames have no frame 0.27 s earlier yet: no model call
    assert [x for x in out if x is not None] == [w]


def test_no_fire_below_threshold():
    out = run([-5] * 30, [(3, 2)] * 30)
    assert all(x is None for x in out)


def test_neighbour_with_more_evidence_wins_and_flicker_does_not_double_count():
    w, nb = (3, 2), (4, 2)
    wells = [nb] * 6 + [w] * 2 + [nb] * 6 + [w] * 2 + [None] * 3 + [nb] * 4
    out = run([5] * 100, wells)
    assert [x for x in out if x is not None] == [nb]  # w had fewer yes frames; nb fired once despite flicker


def test_same_well_fires_again_after_rearm():
    w = (3, 2)
    out = run([5] * 200, [w] * 10 + [None] * 80 + [w] * 10)  # second visit ~3 s later
    assert [x for x in out if x is not None] == [w, w]
