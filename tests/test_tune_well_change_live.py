"""scripts/tune_well_change_live.py queries: dispense at the SPACE press, hover elsewhere, neighbours (synthetic)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tune_well_change_live import queries  # noqa: E402


def test_dispense_hover_and_neighbours():
    t = np.arange(0, 8, 1 / 30)
    H = np.tile(np.array([[30.0, 0, 40], [0, 30, 40], [0, 0, 1]]), (len(t), 1, 1))
    tip = np.full((len(t), 3), np.nan)
    on = (t >= 1.0) & (t < 2.0)
    tip[on] = [40 + 3 * 30, 40 + 2 * 30, 1.0]  # well (3, 2)
    hov = (t >= 5.0) & (t < 6.0)
    tip[hov] = [40 + 6 * 30, 40 + 5 * 30, 1.0]  # well (6, 5), no press
    qs = queries("x", {"H": H, "t": t, "tip": tip}, {0: (3, 2)}, [1.8], 0.4)
    kinds = [(q[0], q[4]) for q in qs]
    assert ((3, 2), "dispense") in kinds and ((6, 5), "hover") in kinds
    assert sum(k == "neighbour" for _, k in kinds) == 8
    assert all(q[6] for q in qs if q[4] == "dispense")  # stop on the labelled well


def test_dry_recording_gives_only_no_liquid_dips():
    t = np.arange(0, 4, 1 / 30)
    H = np.tile(np.array([[30.0, 0, 40], [0, 30, 40], [0, 0, 1]]), (len(t), 1, 1))
    tip = np.full((len(t), 3), np.nan)
    tip[(t >= 1.0) & (t < 2.0)] = [40 + 3 * 30, 40 + 2 * 30, 1.0]
    qs = queries("dry", {"H": H, "t": t, "tip": tip}, {}, [1.8], 0.4, dry=True)
    assert [(q[0], q[4]) for q in qs] == [((3, 2), "dry")]
