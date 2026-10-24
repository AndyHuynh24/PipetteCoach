import cv2
import numpy as np

from pipette_coach.config import load_config
from pipette_coach.localizer import PlateLocalizer
from pipette_coach.localizer.geometry import (
    FLIP_ROWS,
    GRID,
    ROT180,
    grid_error,
    is_mirrored,
    orient,
    project,
    quad_to_H,
)
from pipette_coach.localizer.grid_fit import fit_grid
from pipette_coach.localizer.refine import snap_to_wells, track
from pipette_coach.localizer.rings import find_well_candidates
from tests.synth import plate_H, render_plate

CFG = load_config("localizer")


def _shifted(H, dx, dy, deg=0.0):
    """Move the plate in the image by (dx, dy) px and rotate by deg about its centre."""
    c = project(H, np.array([[5.5, 3.5]]))[0]
    a = np.deg2rad(deg)
    T1 = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T2 = np.array([[1, 0, c[0] + dx], [0, 1, c[1] + dy], [0, 0, 1]])
    return T2 @ R @ T1 @ H


def test_ring_candidates_hit_well_centres():
    H = plate_H(320, 240)
    P, _, _ = find_well_candidates(render_plate(H), CFG)
    d = np.linalg.norm(project(H, GRID)[:, None] - P[None], axis=2).min(1)
    assert (d < 0.2 * 14.5).sum() >= 90


def test_grid_fit_recovers_plate_and_orientation():
    H = plate_H(320, 240, angle_deg=-12)
    g = fit_grid(render_plate(H), CFG)
    assert g is not None and g.frac_on_rings >= CFG["grid_fit"]["confident_frac"]
    Ho = orient(g.H)
    assert grid_error(Ho, H, ignore_labelling=False) < 0.05  # right wells AND A1 top-left
    assert not is_mirrored(Ho)


def test_orient_fixes_mirror_and_180():
    H = plate_H(320, 240)
    for wrong in (H @ FLIP_ROWS, H @ ROT180, H @ FLIP_ROWS @ ROT180):
        assert grid_error(orient(wrong), H, ignore_labelling=False) < 1e-6


def test_orient_keeps_previous_labelling():
    H = plate_H(320, 240)
    prev = H @ ROT180  # e.g. calibrated with A1 at bottom-right
    assert grid_error(orient(H, prev_H=prev), prev, ignore_labelling=False) < 1e-6


def test_snap_from_rough_box():
    H = plate_H(320, 240, angle_deg=5)
    rough = _shifted(H, 4.0, -3.0, deg=1.5)  # ~0.3 pitch off, like a detector box
    s = snap_to_wells(render_plate(H), rough, CFG)
    assert s.ok and grid_error(s.H, H) < 0.05


def test_track_with_half_the_plate_out_of_frame():
    H = plate_H(320, 240)
    moved = _shifted(H, 3.0, 2.0, deg=1.0)
    img = render_plate(moved)[:, :330]  # right ~half of the plate cut off by the frame edge
    t = track(img, H, CFG)
    assert t.ok and grid_error(t.H, moved, ignore_labelling=False) < 0.05


def test_border_hidden_keeps_grid_then_expires():
    import copy

    cfg = copy.deepcopy(CFG)
    cfg["hold"]["max_frames"] = 2
    loc = PlateLocalizer(cfg)  # no detector -> grid-fit fallback
    H = plate_H(320, 240)
    first = loc.update(render_plate(H))
    assert first.status == "detected" and grid_error(first.H, H, ignore_labelling=False) < 0.05
    empty = np.full((480, 640, 3), 98, np.uint8)  # e.g. a hand over the whole plate and its border
    kept = loc.update(empty)
    assert kept.found and np.allclose(kept.H, first.H)  # grid kept exactly
    assert loc.update(empty).found
    assert loc.update(empty).status == "not_found"  # hidden too long: no longer trusted


def test_hand_over_plate_does_not_move_grid():
    loc = PlateLocalizer(CFG)
    H = plate_H(320, 240)
    first = loc.update(render_plate(H))
    for k in range(5):  # a "hand" moving across the wells; the plate border stays where it was
        img = render_plate(H, seed=k + 1)
        cv2.circle(img, (260 + 15 * k, 240), 45, (120, 150, 200), -1)
        fit = loc.update(img)
        assert fit.found and np.array_equal(fit.H, first.H)  # zero jitter


def test_small_real_move_followed_immediately():
    """3 px (0.2 pitch) is enough to risk the wrong well: followed in the very next frame."""
    loc = PlateLocalizer(CFG)
    H = plate_H(320, 240)
    loc.update(render_plate(H))
    moved = _shifted(H, 3.0, 0.0)
    fit = loc.update(render_plate(moved, seed=1))
    assert fit.method == "edge-moved" and grid_error(fit.H, moved, ignore_labelling=False) < 0.05


def test_move_followed_from_wells_when_border_hidden():
    """A hand covers the plate border while pushing the plate: the visible wells still move the grid."""
    from pipette_coach.localizer.geometry import OUTLINE

    loc = PlateLocalizer(CFG)
    H = plate_H(320, 240)
    loc.update(render_plate(H))
    moved = _shifted(H, 3.0, 2.0)
    img = render_plate(moved, seed=1)
    cv2.polylines(img, [np.round(project(moved, OUTLINE)).astype(np.int32)], True, (98, 98, 98), 22)  # border gone
    fit = loc.update(img)
    assert fit.method.startswith("wells-moved") and grid_error(fit.H, moved, ignore_labelling=False) < 0.05


def test_quad_to_H_long_side_is_twelve_columns():
    H = plate_H(320, 240, angle_deg=80)  # plate standing portrait in the image
    from pipette_coach.localizer.geometry import OUTLINE

    quad = project(H, OUTLINE)
    for start in range(4):
        Hq = quad_to_H(np.roll(quad, start, axis=0))
        assert grid_error(Hq, H) < 1e-3


class _StubDetector:
    """Returns a fixed quad, like a YOLO box that only covers part of an occluded plate."""

    def __init__(self, quad):
        self.quad = quad

    def detect(self, bgr):
        return self.quad, 0.95


def _box(H, x0=0.0, x1=1.0):
    """Detector quad covering the fraction [x0, x1] of the plate outline along the 12-well axis."""
    from pipette_coach.localizer.geometry import OUTLINE

    q = project(H, OUTLINE)
    left, right = q[[0, 3]], q[[1, 2]]
    return np.concatenate([left + x0 * (right - left), (left + x1 * (right - left))[::-1]])[[0, 3, 2, 1]]


def test_partial_box_over_covered_plate_is_rebuilt():
    """Right 3 columns hidden under a 'pipette', detector box stops short: full grid recovered."""
    H = plate_H(320, 240, angle_deg=6)
    img = render_plate(H)
    cover = project(H, np.array([[8.5, -2.0], [14.0, -2.0], [14.0, 9.5], [8.5, 9.5]], np.float32))
    cv2.fillConvexPoly(img, np.round(cover).astype(np.int32), (120, 40, 20))  # dark blue pipette head
    fit = PlateLocalizer(CFG, _StubDetector(_box(H, 0.0, 0.72))).detect(img)
    assert fit.found and fit.method == "detector+partial"
    assert grid_error(fit.H, H, ignore_labelling=False) < 0.05


def test_partial_box_two_columns_visible_rebuilt_one_column_refused():
    H = plate_H(320, 240)
    for x_cov, box_x1, ok in ((1.5, 0.3, True), (0.5, 0.25, False)):
        img = render_plate(H)
        cover = project(H, np.array([[x_cov, -2.0], [14.0, -2.0], [14.0, 9.5], [x_cov, 9.5]], np.float32))
        cv2.fillConvexPoly(img, np.round(cover).astype(np.int32), (120, 40, 20))
        fit = PlateLocalizer(CFG, _StubDetector(_box(H, 0.0, box_x1))).detect(img)
        assert fit.found == ok
        if ok:  # ~16 visible wells extrapolated to 96: less precise, but no well can be confused
            assert grid_error(fit.H, H, ignore_labelling=False) < 0.25


def test_orient_with_known_row_direction_turned_plate():
    """Plate turned 90 deg (rows A->H run left->right): A1 must end up bottom-left, never mirrored."""
    H = plate_H(320, 240, angle_deg=-90)  # canonical x (columns) points up, y (rows) points right
    for wrong in (H, H @ ROT180, H @ FLIP_ROWS):
        Ho = orient(wrong, rows_toward="right")
        assert grid_error(Ho, H, ignore_labelling=False) < 1e-6
        a1, a12, h1 = project(Ho, np.array([[0, 0], [11, 0], [0, 7]], np.float32))
        assert a12[1] < a1[1] and h1[0] > a1[0]  # columns go up, rows go right


def test_filled_wells_found_by_edge_alignment():
    """Milky / dyed wells have no dark centre (ring filter fails); edge-based whole-grid alignment still works."""
    from pipette_coach.localizer.geometry import OUTLINE

    H = plate_H(320, 240, angle_deg=4)
    img = render_plate(H)
    p = 14.5
    for k, (x, y) in enumerate(project(H, GRID)):
        col = (235, 235, 235) if k % 3 else (40, 40, 210)  # milky + red dye
        cv2.circle(img, (int(round(x)), int(round(y))), int(0.33 * p), col, -1, cv2.LINE_AA)
    rough = project(_shifted(H, 3.0, -2.0, deg=1.0), OUTLINE)  # detector box ~0.25 pitch off
    fit = PlateLocalizer(CFG, _StubDetector(rough)).detect(img)
    assert fit.found and fit.method == "detector+align"
    assert grid_error(fit.H, H, ignore_labelling=False) < 0.1


def test_measured_plate_box_lands_on_plate_edges():
    from pipette_coach.localizer.geometry import OUTLINE, project
    from pipette_coach.localizer.plate_edges import measure_plate_box

    cfg = load_config("localizer")
    H = plate_H(320, 240, angle_deg=6.0)
    img = render_plate(H)
    true = project(H, OUTLINE)
    rough = (true - true.mean(0)) * 1.08 + true.mean(0) + [6.0, -4.0]  # loose, shifted box like the detector's
    box, scores, _ = measure_plate_box(img, rough, 14.5, cfg)
    assert np.abs(box - true).max() < 1.0, (box, true)
    assert (scores > 0).all()


def test_plate_box_parallel_constraint_and_fusion_ignore_an_occluder():
    from pipette_coach.localizer.geometry import OUTLINE, project
    from pipette_coach.localizer.plate_edges import fuse_boxes, measure_plate_box

    cfg = load_config("localizer")
    H = plate_H(320, 240, angle_deg=-4.0)
    true = project(H, OUTLINE)
    rough = (true - true.mean(0)) * 1.06 + true.mean(0)
    meas = []
    for k in range(5):
        img = render_plate(H, seed=k)
        x = 360 + 40 * k  # a dark "pipette" bar over part of the bottom side, at a different place each frame
        cv2.rectangle(img, (x, 250), (x + 70, 479), (120, 40, 20), -1)
        meas.append(measure_plate_box(img, rough, 14.5, cfg))
    fused = fuse_boxes(meas, cfg)
    assert fused is not None
    box, support = fused
    clean, _, _ = measure_plate_box(render_plate(H), rough, 14.5, cfg)  # same plate, nothing over it
    d = np.linalg.norm(box[:, None] - clean[None], axis=2).min(1)
    assert d.max() < 0.5, (box, clean)  # the occluder does not move the fused box
    assert np.linalg.norm(box[:, None] - true[None], axis=2).min(1).max() < 2.0  # renderer rounds corners to px


def test_complete_box_rebuilds_a_hidden_side_from_the_plate_size():
    from pipette_coach.localizer.plate_edges import box_size, complete_box

    cfg = load_config("localizer")
    true = np.array([[100.0, 100.0], [306.0, 100.0], [306.0, 238.0], [100.0, 238.0]])
    size = box_size(true)
    seen = true.copy()
    seen[[1, 2], 0] -= 40  # right side measured on a pipette parked over the plate
    support = np.array([1.0, 1.0, 0.0, 1.0])  # side 2 = corner 1 -> corner 2 = right side: never measured
    fixed, rebuilt = complete_box(seen, support, size, None, cfg)
    assert rebuilt == [2]
    assert np.abs(fixed - true).max() < 1e-6
    fixed2, rebuilt2 = complete_box(seen, support, None, size[1] / size[0], cfg)  # no own size: ratio
    assert rebuilt2 == [2] and np.abs(fixed2 - true).max() < 1e-6


def test_colour_veto_rejects_a_pipette_boundary_as_plate_edge():
    import copy

    from pipette_coach.localizer.geometry import OUTLINE, project
    from pipette_coach.localizer.plate_edges import measure_plate_box

    cfg = load_config("localizer")
    H = plate_H(320, 240, angle_deg=0.0)
    true = project(H, OUTLINE)
    img = render_plate(H)
    x_bar = int(true[:, 0].max() - 12)  # blue pipette parked over the right side, the whole length
    cv2.rectangle(img, (x_bar, 0), (x_bar + 60, 479), (130, 40, 20), -1)
    rough = (true - true.mean(0)) * 1.05 + true.mean(0)
    cfg["plate_edges"]["use_colour"] = True  # optional feature (off by default: clear plates look like the bench)
    no_col = copy.deepcopy(cfg)
    no_col["plate_edges"]["use_colour"] = False
    box_nc, sc_nc, lines_nc = measure_plate_box(img, rough, 14.5, no_col)
    box_c, sc_c, _ = measure_plate_box(img, rough, 14.5, cfg)
    right = int(np.argmax([(a[0] + b[0]) / 2 for a, b in lines_nc]))
    assert sc_nc[right] > 0  # edge alone takes the plate | pipette boundary as the plate edge
    assert abs((lines_nc[right][0][0] + lines_nc[right][1][0]) / 2 - x_bar) < 2
    assert sc_c[right] == 0  # colour veto: pipette colour outside -> side hidden in this frame
    assert all(sc_c[k] > 0 for k in range(4) if k != right)


def test_box_from_wells_rebuilds_a_side_hidden_under_a_pipette():
    from pipette_coach.localizer.geometry import GRID, OUTLINE, project
    from pipette_coach.localizer.wells_box import box_from_wells, fit_lattice, plate_size_pitch, pool_points

    cfg = load_config("localizer")
    H = plate_H(320, 240, angle_deg=7.0)
    H[2, :2] = [2e-5, -1e-5]  # a little perspective
    true = project(H, OUTLINE)
    rng = np.random.default_rng(0)
    wells = project(H, GRID)
    H_all, ij_all, _ = fit_lattice(wells, (11, 20))
    size = plate_size_pitch(H_all, true, ij_all)  # clean window: all edges + all wells
    assert abs(size[0] - 14.2) < 0.05 and abs(size[1] - 9.5) < 0.05
    # another window: columns 10-12 hidden under a pipette, the right edge never measured, wells seen in 3 frames
    vis = wells[GRID[:, 0] < 9]
    frames = [vis + rng.normal(0, 0.2, vis.shape) for _ in range(3)]
    P = pool_points(frames, 14.5)
    H_rel, ij, _ = fit_lattice(P, (11, 20))
    edge_box = true.copy()
    for k, j in ((1, 0), (2, 3)):  # measured box stops at the pipette (corners slide along top/bottom sides) ...
        edge_box[k] += (true[j] - true[k]) * 30 / np.linalg.norm(true[j] - true[k])
    support = np.array([1.0, 1.0, 0.0, 1.0])  # ... and that side (corner 1 -> corner 2) was never measured
    box, src, _ = box_from_wells(H_rel, ij, edge_box, support, size, (12, 8), cfg)
    assert sorted(src.values()).count("size") == 1
    assert np.linalg.norm(box[:, None] - true[None], axis=2).min(1).max() < 0.6
    # same, but the pipette boundary is a clean straight edge seen in EVERY frame (support 1): rim rule catches it
    box2, src2, _ = box_from_wells(H_rel, ij, edge_box, np.ones(4), size, (12, 8), cfg)
    assert sorted(src2.values()).count("size") == 1
    assert np.linalg.norm(box2[:, None] - true[None], axis=2).min(1).max() < 0.6
    # a window with the pipette boundary taken as an edge must not count toward the plate size
    H_v, ij_v, _ = fit_lattice(pool_points(frames, 14.5), (11, 20))
    bad = plate_size_pitch(H_v, edge_box, ij_v)
    assert np.isnan(bad[0]) and abs(bad[1] - 9.5) < 0.05  # long axis rejected (wells hidden), short axis fine


def test_grid_forms_12x8_and_snaps_detections_dropping_off_grid_ones():
    from pipette_coach.localizer.geometry import GRID, OUTLINE, grid_error, project
    from pipette_coach.localizer.wells_box import box_from_wells, fit_lattice, grid_from_lattice, snap_frame

    cfg = load_config("localizer")
    H = plate_H(320, 240, angle_deg=-5.0)
    true_box = project(H, OUTLINE)
    wells = project(H, GRID)
    vis = wells[GRID[:, 0] < 8]  # columns 9-12 hidden
    H_rel, ij, _ = fit_lattice(vis, (11, 20))
    box, src, info = box_from_wells(H_rel, ij, true_box, np.ones(4), (14.2, 9.5), (12, 8), cfg)
    Hg = grid_from_lattice(H_rel, ij, info["rect"], info["u_long"], (12, 8))
    assert grid_error(Hg, H) < 0.05  # labelling-free: the 12x8 grid lands on the true wells, hidden ones too
    # a frame: visible wells slightly off (perspective/noise), plus a doubled column half a well over (tips)
    rng = np.random.default_rng(1)
    dets = np.concatenate([vis + rng.normal(0, 0.6, vis.shape), project(H, GRID[GRID[:, 0] == 7] + [0.5, 0])])
    Hf, matched, pos = snap_frame(Hg, dets)
    assert matched.sum() == len(vis)  # every visible well matched once; the doubled column dropped
    d = np.linalg.norm(pos[:, None] - wells[None], axis=2)  # labelling (A1 corner) is orient()'s job
    assert d.min(1).max() < 3.0 and len(set(d.argmin(1))) == 96  # every true well covered exactly once


def test_snap_frame_on_a_narrow_strip_does_not_fan_out_hidden_wells():
    from pipette_coach.localizer.geometry import GRID, project
    from pipette_coach.localizer.wells_box import snap_frame

    H = plate_H(320, 240, angle_deg=3.0)
    wells = project(H, GRID)
    rng = np.random.default_rng(2)
    strip = GRID[:, 0] < 3  # only columns 1-3 visible (pipette over the rest)
    dets = wells[strip] + rng.normal(0, 0.7, (strip.sum(), 2))
    H0 = H @ np.array([[1, 0, 0.12], [0, 1, -0.1], [0, 0, 1]])  # starting grid a little off
    Hf, matched, pos = snap_frame(H0, dets)
    assert matched.sum() == strip.sum()
    assert np.linalg.norm(pos[~strip] - wells[~strip], axis=1).max() < 3.0  # hidden wells stay on the lattice


def test_plate_grid_helpers():
    from pipette_coach.localizer.plate_grid import grow, window_at

    q = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    g = grow(q, 1.0, 1.0)
    assert np.allclose(np.linalg.norm(g - q, axis=1), 1.0)
    wins = [{"start": 0, "end": 30}, {"start": 30, "end": 72}]
    assert window_at(wins, 10) is wins[0] and window_at(wins, 30) is wins[1] and window_at(wins, 99) is wins[1]


def _visible_case(H, cols, rows):
    """Lattice result + visible box for a plate (canonical -> image H) with only some wells visible."""
    from pipette_coach.localizer.geometry import GRID, OUTLINE, project
    from pipette_coach.localizer.wells_box import fit_lattice

    m = np.isin(GRID[:, 0], cols) & np.isin(GRID[:, 1], rows)
    H_rel, ij, _ = fit_lattice(project(H, GRID[m]), (11, 20))
    # visible box: the plate outline cut where the hidden wells start (half a spacing past the last visible well)
    xs = np.clip(OUTLINE[:, 0], -1.6, max(cols) + 0.5 if max(cols) < 11 else 12.6)
    ys = np.clip(OUTLINE[:, 1], -1.25, max(rows) + 0.5 if max(rows) < 7 else 8.25)
    vis = project(H, np.stack([xs, ys], 1))
    return {"H_rel": H_rel, "ij": ij, "scale_fit": 1.0}, np.array([*vis.min(0), *vis.max(0)])


def test_rebuild_from_visible_completes_length_and_width_and_refuses_to_guess():
    from pipette_coach.localizer.geometry import OUTLINE, grid_error, project
    from pipette_coach.localizer.wells_first import rebuild_from_visible

    cfg = load_config("localizer")
    cfg["plate_edges"]["plate_size_pitch"] = [14.2, 9.5]  # the synthetic plate's outline (OUTLINE)
    for angle in (0.0, 90.0):  # plate lying sideways, and upright
        H = plate_H(500, 400, angle_deg=angle)
        true = project(H, OUTLINE)
        # columns 10-12 hidden (length cut): 9 wells along the long side -> length; rebuilt from its real end
        r, vis = _visible_case(H, range(9), range(8))
        out = rebuild_from_visible(r, vis, cfg)
        assert out is not None and out["counts"] in ([9, 8], [8, 9])
        assert np.linalg.norm(out["outline"][:, None] - true[None], axis=2).min(1).max() < 1.0
        assert grid_error(out["H_grid"], H) < 0.05
        # rows E-H hidden (width cut): all 12 columns visible -> long side known; width rebuilt from length
        r, vis = _visible_case(H, range(12), range(4))
        out = rebuild_from_visible(r, vis, cfg)
        assert out is not None and grid_error(out["H_grid"], H) < 0.05
        # only 6 x 6 wells visible: length vs width cannot be told -> no guess
        r, vis = _visible_case(H, range(6), range(6))
        assert rebuild_from_visible(r, vis, cfg) is None


def test_plausible_outline_accepts_plates_and_rejects_skewed_or_oversized_ones():
    from pipette_coach.localizer.wells_first import plausible_outline

    plate = np.array([[100, 100], [239, 100], [239, 195], [100, 195]], float)  # 13.9 x 9.5 at 10 px
    box = np.array([100, 100, 239, 195], float)
    assert plausible_outline(plate, box)
    upright_tilted = np.array([[100, 100], [195, 110], [185, 245], [92, 238]], float)
    assert plausible_outline(upright_tilted, np.array([92, 100, 195, 245], float))
    fan = np.array([[100, 100], [239, 100], [330, 300], [60, 300]], float)  # skewed extrapolation
    assert not plausible_outline(fan, box)
    assert not plausible_outline(plate * 3, box)  # far bigger than the visible plate
    square = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], float)  # not plate-shaped
    assert not plausible_outline(square, box)


def test_track_grid_follows_a_small_move_from_a_few_visible_wells_and_keeps_identity():
    from pipette_coach.localizer.geometry import GRID, grid_error, project
    from pipette_coach.localizer.wells_first import track_grid

    cfg = load_config("localizer")
    H0 = plate_H(320, 240, angle_deg=4.0)
    H1 = plate_H(323, 238, angle_deg=5.0)  # the plate moved a little
    vis = (GRID[:, 0] >= 8) & (GRID[:, 1] <= 3)  # only the top-right 4 x 4 wells visible (pipette over the rest)
    dets = project(H1, GRID[vis])
    out = track_grid(H0, dets, cfg)
    assert out is not None and out[1] == vis.sum()
    assert grid_error(out[0], H1, ignore_labelling=False) < 0.05  # whole grid incl. hidden wells, same A1
    assert track_grid(H0, dets[:3], cfg) is None  # too few wells: no guess
