"""pipette_coach/liquid_net.py and the grid settling / clip windows of the liquid scripts (synthetic, no video)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_liquid_clips as blc  # noqa: E402
import build_liquid_dataset as bld  # noqa: E402

from pipette_coach.liquid_net import LiquidNet, S, flip, press_well, to_input  # noqa: E402


def test_one_output_per_well():
    x = torch.zeros(2, 6, 8 * S, 12 * S)
    assert LiquidNet(pretrained=False)(x).shape == (2, 8, 12)


def test_press_order_and_input():
    assert press_well(0) == (11, 7) and press_well(7) == (11, 0) and press_well(8) == (10, 7)
    assert press_well(95) == (0, 0)
    img = np.zeros((8 * S, 12 * S, 3), np.uint8)
    assert to_input(img, img).shape == (6, 8 * S, 12 * S)


def test_flip_moves_label_with_image():
    now = np.zeros((8 * S, 12 * S, 3), np.uint8)
    now[2 * S : 3 * S, 5 * S : 6 * S] = 255  # well (col 5, row 2)
    t = np.zeros((8, 12), np.float32)
    t[2, 5] = 1
    n2, _, t2 = flip(now, now.copy(), t, True, True)
    r, c = np.argwhere(t2 == 1)[0]
    assert (r, c) == (5, 6) and n2[r * S + S // 2, c * S + S // 2].max() == 255


def test_snap_prefers_grid_with_wells_inside_border_outside():
    R = np.zeros((8 + 2 * bld.M, 12 + 2 * bld.M))
    R[bld.M + 1 : bld.M + 9, bld.M : bld.M + 12] = 1  # wells sit one row lower than the grid says
    sc = bld.snap_scores(R)
    assert max(sc, key=sc.get) == (0, 1)


def test_stretches_split_at_jumps():
    H0 = np.eye(3) * 1.0
    H0[0, 0] = H0[1, 1] = 30
    H1 = bld.shift(H0, 0, 1)  # one well down
    runs = bld.stretches([H0, H0, None, H0, H1, H1], 0.35)
    assert runs == [[0, 1], [3], [4, 5]]


def test_windows_clip_to_video():
    w = blc.windows([{"frame": 10, "t": 0.33}, {"frame": 290, "t": 9.7}], 30.0, 300, 1.0)
    assert (w[0]["f0"], w[0]["f1"]) == (0, 40) and (w[1]["f0"], w[1]["f1"]) == (260, 299)


def test_crop_well_centres_the_well_and_pads_edges():
    from pipette_coach.liquid_net import CROP, LiquidCropNet, crop_well

    plate = np.zeros((8 * S, 12 * S, 3), np.uint8)
    plate[0:S, 0:S] = 200  # well A1 (col 0, row 0)
    c = crop_well(plate, 0, 0)
    assert c.shape == (CROP * S, CROP * S, 3)
    assert c[S + S // 2, S + S // 2, 0] == 200  # A1 in the middle cell
    assert crop_well(plate, 11, 7).shape == c.shape
    x = torch.zeros(2, 6, CROP * S, CROP * S)
    assert LiquidCropNet(pretrained=False)(x).shape == (2,)
