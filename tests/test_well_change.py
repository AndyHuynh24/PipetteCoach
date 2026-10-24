import cv2
import numpy as np

from pipette_coach.localizer.geometry import GRID, OUTLINE, project
from pipette_coach.well_change import DispenseCheck, WellChange
from tests.synth import plate_H

H = plate_H(320, 240, pitch=30, angle_deg=5)
PITCH = 30


def frame(filled=(), dye=None, gain=1.0, hand=None, seed=0):
    """Empty wells: dark bottom with a bright wall crescent. Filled: flat, no crescent (clear) or tinted (dye)."""
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), (92, 98, 104), np.uint8)
    cv2.fillConvexPoly(img, np.round(project(H, OUTLINE)).astype(np.int32), (150, 153, 155))
    for k, (x, y) in enumerate(project(H, GRID)):
        c, r = int(round(x)), int(round(y))
        cv2.circle(img, (c, r), 13, (110, 112, 114), -1, cv2.LINE_AA)
        if (k % 12, k // 12) in filled:
            col = (120, 122, 124) if dye is None else dye
            cv2.circle(img, (c, r), 12, col, -1, cv2.LINE_AA)
        else:
            cv2.ellipse(img, (c - 3, r), (8, 11), 0, 100, 260, (215, 218, 220), 3, cv2.LINE_AA)
    if hand is not None:  # skin-coloured blob over a well and its neighbours
        cv2.circle(img, tuple(np.round(project(H, np.array([hand], np.float32))[0]).astype(int)), 40,
                   (120, 160, 220), -1)  # fmt: skip
    img = np.clip(img * gain + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    return img


def run(dye=None):
    wc = WellChange()
    for i in range(120):  # 12 s at 10 fps; the tip/hand is over well (3, 2) during 5-6 s; filled from 6 s on
        t = i / 10
        hand = (3, 2) if 5 <= t < 6 else None
        wc.update(t, frame(filled={(3, 2)} if t >= 6 else (), dye=dye, gain=1.0 if t < 6 else 0.9, hand=hand, seed=i),
                  H)  # fmt: skip
    return wc


def test_clear_liquid_is_seen_and_neighbours_are_not():
    wc = run()
    hit, other = wc.score((3, 2), 5, 6), wc.score((4, 2), 5, 6)
    assert hit["filled"] is True and hit["tex"] < 0.8
    assert other["filled"] is False


def test_dye_is_not_taken_for_a_hand():
    wc = run(dye=(60, 60, 200))  # red liquid: must still count as "seen", and move the colour
    hit = wc.score((3, 2), 5, 6)
    assert hit["filled"] is True and hit["de"] > 5


def test_dispense_check_gives_a_verdict_after_the_tip_leaves():
    chk, got = DispenseCheck(), []
    chk.watch(0, (3, 2), 5.0)
    for i in range(100):
        t = i / 10
        tip = (3, 2) if 5 <= t < 6 else None
        got += chk.update(t, frame(filled={(3, 2)} if t >= 6 else (), seed=i), H, tip)
    assert len(got) == 1 and got[0][0] == 0 and got[0][1]["filled"] is True
