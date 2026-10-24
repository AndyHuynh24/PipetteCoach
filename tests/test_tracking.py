import cv2
import numpy as np

from pipette_coach.localizer.geometry import GRID, project
from pipette_coach.protocol.wells import all_wells, index_well, parse_well, well_index
from pipette_coach.tracking import plate_view as pv
from pipette_coach.tracking.tip_geom import tip_end
from tests.synth import plate_H


def test_well_index_roundtrip():
    assert [well_index(w) for w in all_wells()] == list(range(96))
    assert well_index(parse_well("A12")) == 11 and well_index(parse_well("H1")) == 84
    assert all(well_index(index_well(i)) == i for i in range(96))


def test_plate_view_puts_every_well_at_its_fixed_pixel():
    H = plate_H(320, 240, angle_deg=12.0)
    img = np.zeros((480, 640, 3), np.uint8)
    for x, y in project(H, GRID):
        cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), 3 * 16, (255, 255, 255), -1, cv2.LINE_AA, 4)
    v = cv2.cvtColor(pv.rectify(img, H), cv2.COLOR_BGR2GRAY).astype(np.float32)
    for k, (x, y) in enumerate(pv.NODES):
        patch = v[int(y) - 6 : int(y) + 7, int(x) - 6 : int(x) + 7]
        cy, cx = np.unravel_index(np.argmax(cv2.GaussianBlur(patch, (0, 0), 2)), patch.shape)
        assert abs(cx - 6) <= 1 and abs(cy - 6) <= 1, k
    assert pv.nearest_well(pv.NODES[13] + [3, -2]) == 13
    assert pv.nearest_well(pv.NODES[13] + [7, 7]) is None  # the gap between four wells


def test_tip_end_is_the_deepest_point_of_an_object_entering_from_outside():
    mask = np.zeros((pv.H, pv.W), np.uint8)
    tip = pv.to_view(np.array([[4.0, 3.0]]))[0]  # well D5
    cv2.line(mask, (pv.W - 1, pv.H - 1), (int(tip[0]), int(tip[1])), 1, 5)  # shaft from the bottom-right corner
    cv2.circle(mask, (30, 30), 6, 1, -1)  # a blob inside, not entering: ignored
    r = tip_end(mask)
    assert r is not None and np.linalg.norm(r[0] - tip) < 3.5
    assert pv.nearest_well(r[0]) == well_index(parse_well("D5"))


def test_trace_tip_follows_a_faint_tip_past_the_shaft_end():
    from pipette_coach.tracking.tip_geom import shaft_axis, trace_tip

    rng = np.random.default_rng(0)
    diff = np.abs(rng.normal(3, 1, (pv.H, pv.W))).astype(np.float32)  # background noise
    entry, shaft_end, tip = np.array([223.0, 167.0]), np.array([150.0, 110.0]), np.array([120.0, 86.6])
    mask = np.zeros((pv.H, pv.W), np.uint8)
    cv2.line(mask, tuple(entry.astype(int)), tuple(shaft_end.astype(int)), 1, 7)
    cv2.line(diff, tuple(shaft_end.astype(int)), tuple(tip.astype(int)), 12.0, 1)  # faint 1-px tip, diff 12
    u = shaft_axis(mask, shaft_end)
    cfg = {"trace_band_px": 1.5, "trace_k_mad": 4.0, "trace_gap_px": 3.0, "trace_max_pitch": 3.0}
    t = trace_tip(diff, shaft_end, u, cfg)
    assert np.linalg.norm(t - tip) < 3.0


def test_near_tips_picks_the_object_touching_the_clicked_tip():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_pipette_yolo_dataset import near_tips

    m = np.zeros((pv.H, pv.W), np.uint8)
    cv2.line(m, (223, 160), (120, 90), 1, 5)  # pipette
    cv2.circle(m, (20, 20), 8, 1, -1)  # a hand blob elsewhere
    keep = near_tips(m, np.array([[121.0, 91.0]]), 1.5 * pv.PX)
    assert keep[90, 120] and keep[155, 215] and not keep[20, 20]


def test_wells_cluster_box_finds_the_plate_and_skips_stray_wells():
    from pipette_coach.localizer.live import wells_cluster_box

    rng = np.random.default_rng(0)
    plate = np.array([[100 + 15 * c, 50 + 15 * r] for r in range(8) for c in range(12)], float)
    stray = rng.uniform([400, 300], [600, 400], (15, 2))  # scattered false wells far from the plate
    box = wells_cluster_box(np.vstack([plate + rng.normal(0, 0.5, plate.shape), stray]), pitch=14.5)
    assert box is not None and np.allclose(box, [100 - 13.5, 50 - 13.5, 265 + 13.5, 155 + 13.5], atol=2)
    assert wells_cluster_box(plate[:20], pitch=14.5) is None  # too few wells
    assert wells_cluster_box(plate * 3, pitch=14.5) is None  # spacing far from the well net's size
