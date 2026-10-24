import numpy as np

from pipette_coach.config import load_config
from pipette_coach.localizer.geometry import GRID, grid_error, project
from pipette_coach.localizer.live import PlateTracker


def _H(dx=0.0):
    """Canonical grid -> image: 30 px spacing, shifted dx spacings along the 12-well axis."""
    return np.array([[30.0, 0, 200 + 30 * dx], [0, 30.0, 150], [0, 0, 1]])


def test_anchor_rejects_slipped_grid():
    tr = PlateTracker(load_config("localizer"), det=None, net=None)
    tr.anchor = _H()
    dets = project(_H(), GRID[GRID[:, 0] < 6])  # half the plate visible (a hand over the rest)
    good = tr._from_anchor((_H(0.1), 48), dets)
    assert grid_error(good[0], _H(0.1), ignore_labelling=False) < 1e-6  # small move: kept
    fixed = tr._from_anchor((_H(1.0), 48), dets)  # read one column off: refitted from the anchor
    assert fixed is not None and grid_error(fixed[0], _H(), ignore_labelling=False) < 0.05
    tr.use_anchor = False
    assert grid_error(tr._from_anchor((_H(1.0), 48), dets)[0], _H(1.0), ignore_labelling=False) < 1e-6


def test_box_check_rejects_grid_off_the_plate():
    tr = PlateTracker(load_config("localizer"), det=None, net=None)
    tr.box = np.array([180.0, 130, 550, 380])  # outer well edges of _H(): x 185-545, y 135-375
    assert tr._box_ok(_H())
    assert not tr._box_ok(_H(1.0))  # one column off: sticks out 0.83 spacing
    tr.box = None
    assert tr._box_ok(_H(1.0))  # no box: no check


def test_anchor_follows_pushed_plate_with_box():
    tr = PlateTracker(load_config("localizer"), det=None, net=None)
    tr.box = np.array([180.0, 130, 550, 380])
    tr._set_anchor(_H())
    tr.box = tr.box + [36, 0, 36, 0]  # plate pushed 1.2 spacings along the rows
    dets = project(_H(1.2), GRID[GRID[:, 0] < 6])  # half the plate visible
    got = tr._from_anchor((_H(1.0), 48), dets)  # tracking read the neighbouring column
    assert got is not None and grid_error(got[0], _H(1.2), ignore_labelling=False) < 0.05
    assert grid_error(tr.anchor, _H(1.2), ignore_labelling=False) < 0.05  # the anchor moved with the plate


def test_steady_output_averages_jitter_follows_moves():
    tr = PlateTracker(load_config("localizer"), det=None, net=None)
    tr._steadied(_H())
    small = tr._steadied(_H(0.06))  # jitter: a quarter of the way
    assert abs(grid_error(small, _H(), ignore_labelling=False) - 0.015) < 0.005
    big = tr._steadied(_H(1.0))  # pushed: followed at once
    assert grid_error(big, _H(1.0), ignore_labelling=False) < 1e-6
    assert tr._steadied(None) is None and tr.H_show is None


def test_plate_edges_put_a_grid_read_one_column_off_back():
    import cv2

    cfg = load_config("localizer")
    long_p, short_p = cfg["plate_edges"]["plate_size_pitch"]
    ru, rv = (long_p - 11) / 2, (short_p - 7) / 2
    img = np.full((600, 800, 3), 40, np.uint8)  # dark bench, light plate (its footprint drawn from _H())
    corners = project(_H(), np.array([[-ru, -rv], [11 + ru, -rv], [11 + ru, 7 + rv], [-ru, 7 + rv]]))
    cv2.fillConvexPoly(img, np.round(corners).astype(np.int32), (190, 190, 190))
    tr = PlateTracker(cfg, det=None, net=None)
    tr.box = np.array([*corners.min(0) - 5, *corners.max(0) + 5])
    assert tr._edge_shift(img, _H()) == (0, 0)
    assert tr._edge_shift(img, _H(1.0)) == (-1, 0)  # grid one column right of the plate: move it back
