"""Visible-well rule for the well-net training labels (pipette_coach/localizer/well_labels.py)."""

import cv2
import numpy as np

from pipette_coach.localizer import well_labels as wl
from pipette_coach.localizer.geometry import GRID, project

H = cv2.findHomography(np.float32([[0, 0], [11, 0], [11, 7], [0, 7]]),
                       np.float32([[200, 150], [860, 150], [860, 570], [200, 570]]), 0)[0]  # fmt: skip


def plate(rng, shift=(0, 0)):
    f = np.full((720, 1080, 3), 90, np.uint8)
    for x, y in project(H, GRID) + shift:
        cv2.circle(f, (int(x), int(y)), 22, (200, 200, 200), 3)
    return np.clip(f.astype(int) + rng.integers(-4, 5, f.shape), 0, 255).astype(np.uint8)


def test_pipette_over_a_well_is_a_change_small_misalignment_is_not():
    rng = np.random.default_rng(0)
    refs = [wl.flat_lab(plate(rng), H) for _ in range(5)]
    f = plate(rng, shift=(1.5, -1))  # grid ~1.5 px off: not a change
    cv2.line(f, (530, 300), (560, 260), (250, 250, 250), 6)  # thin white tip over one well
    ch = wl.change(wl.flat_lab(f, H), refs)
    covered = int(np.argmax(ch))
    assert ch[covered] >= wl.THRESHOLD
    assert (np.delete(ch, covered) < wl.THRESHOLD).mean() > 0.97


def test_reference_frames_same_place_and_window():
    T = np.array([[1, 0, 1.0], [0, 1, 0], [0, 0, 1]])  # plate moved one column
    items = [(0, H, 0.0), (1, H, 20.0), (2, H, 50.0), (3, H @ T, 21.0)]
    assert wl.reference_frames(items, 0) == [0, 1]
    assert wl.reference_frames(items, 3) == [3]
