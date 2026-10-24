import numpy as np

from pipette_coach.tips import (
    negative_candidates,
    parse_cvat_plates,
    parse_cvat_points,
    plate_crop_box,
    pose_lines,
    random_crop_box,
    split_of,
)

XML = """<?xml version="1.0"?><annotations>
<image id="0" name="frame_000000" width="100" height="50"></image>
<image id="1" name="frame_000001" width="100" height="50">
  <points label="Tip" source="manual" occluded="0" points="10.0,20.0" z_order="0"></points></image>
<image id="2" name="frame_000002" width="100" height="50">
  <points label="tip" source="manual" occluded="0" points="1,2;3,4" z_order="0"></points>
  <points label="other" source="manual" occluded="0" points="5,5" z_order="0"></points></image>
<image id="3" name="frame_000003" width="100" height="50">
  <points label="Tip" source="manual" occluded="0" points="7.5,8.5;7.5,8.5" z_order="0"></points>
  <box label="Plate" source="manual" occluded="0" xtl="1" ytl="2" xbr="30" ybr="40" z_order="0"></box></image>
</annotations>"""


def test_parse_cvat_points(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(XML)
    tips, n = parse_cvat_points(p)
    assert n == 4
    assert tips == {1: [(10.0, 20.0)], 2: [(1.0, 2.0), (3.0, 4.0)], 3: [(7.5, 8.5)]}  # double click -> one tip
    assert parse_cvat_plates(p) == {3: (1.0, 2.0, 30.0, 40.0)}
    assert parse_cvat_plates(p, drop=(3,)) == {}


def test_plate_mask_box_drops_stray_blobs(tmp_path):
    # 40 x 30 mask at (100, 200): a 30 x 20 block (rows 5-24, cols 5-34) + a 2 x 2 stray blob at rows 0-1, cols 0-1
    m = np.zeros((30, 40), np.uint8)
    m[5:25, 5:35] = 1
    m[0:2, 0:2] = 1
    flat, runs, val, n = m.ravel(), [], 0, 0
    for v in flat:
        if v != val:
            runs.append(n)
            val, n = v, 0
        n += 1
    runs.append(n)
    xml = (
        f'<annotations><image id="0" name="f" width="300" height="300"><mask label="Plate" '
        f'rle="{", ".join(map(str, runs))}" left="100" top="200" width="40" height="30"/></image></annotations>'
    )
    p = tmp_path / "m.xml"
    p.write_text(xml)
    assert parse_cvat_plates(p) == {0: (105.0, 205.0, 135.0, 225.0)}


def test_split_and_negatives():
    v = {"val_from": 10, "test_from": 20, "skip": [[0, 1]]}
    assert [split_of(t, v) for t in (0, 10, 25)] == ["train", "val", "test"]
    c = negative_candidates({100}, 300, 30.0, 1.0, v, 1000)
    assert all(abs(f - 100) > 30 for f in c)  # >= 1 s from the label
    assert all(f >= 30 for f in c)  # skip range [0, 1) s
    assert len(negative_candidates({100}, 300, 30.0, 1.0, v, 5)) == 5


def test_crops_and_lines():
    box = plate_crop_box(np.array([[100, 100], [300, 100], [300, 200], [100, 200]]), 0.25, 1000, 1000)
    assert box == (50, 75, 350, 225)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x0, y0, x1, y1 = random_crop_box(rng, (0.5, 1.0), 400, 200, keep=(390, 10))
        assert 0 <= x0 and x1 <= 400 and 0 <= y0 and y1 <= 200 and x1 - x0 == y1 - y0
        assert x0 <= 390 < x1 and y0 <= 10 < y1
    (ln,) = pose_lines([(150, 100)], (100, 50, 300, 150), 0.1)
    c, x, y, bw, bh, kx, ky, vis = ln.split()
    assert (float(x), float(y), float(bw), float(bh), float(kx), vis) == (0.25, 0.5, 0.1, 0.2, 0.25, "2")
    assert pose_lines([(50, 100)], (100, 50, 300, 150), 0.1) is None
    assert pose_lines([], (0, 0, 10, 10), 0.1) == []


def test_well_of():
    from pipette_coach.tips import well_of

    H = np.array([[10.0, 0, 100], [0, 10.0, 50], [0, 0, 1]])  # 10 px per well spacing, A1 at (100, 50)
    assert well_of(H, (100, 50)) == (0, 0)
    assert well_of(H, (213, 118)) == (11, 7)  # 0.3 / 0.2 spacing off the centre of the last well
    assert well_of(H, (105, 55)) is None  # between wells
    assert well_of(H, (300, 50)) is None  # off the plate


def test_dwell_detector():
    from pipette_coach.tips import DwellDetector

    d = DwellDetector(min_s=0.4, gap_s=0.25)
    ev = [d.update(i / 10, w) for i, w in enumerate([(1, 1)] * 3 + [(2, 2)] * 6)]
    assert [e for e in ev if e] == [(2, 2)]  # 0.3 s on (1, 1) is a pass-over; 0.5 s on (2, 2) is a dispense
    d = DwellDetector(min_s=0.4, gap_s=0.25)
    seq = [(3, 3), None, (3, 3), (3, 3), None, (3, 3), (3, 3), (3, 3)]  # short misses do not end the visit
    assert [e for e in (d.update(i / 10, w) for i, w in enumerate(seq)) if e] == [(3, 3)]
    d2 = DwellDetector(min_s=0.4, gap_s=0.25, refractory_s=1.5)
    seq2 = [(4, 4)] * 5 + [None] * 5 + [(4, 4)] * 5  # tip out of view 0.5 s, back on the same well: same visit
    assert sum(1 for i, w in enumerate(seq2) if d2.update(i / 10, w)) == 1
    d3 = DwellDetector(min_s=0.4, gap_s=0.25, refractory_s=1.5)
    seq3 = [(4, 4)] * 5 + [None] * 30 + [(4, 4)] * 5  # back after 3 s (new aspiration): a second dispense
    assert sum(1 for i, w in enumerate(seq3) if d3.update(i / 10, w)) == 2
    d4 = DwellDetector(min_s=0.4, gap_s=0.25, refractory_s=1.5, trip_gap_s=None)
    seq4 = [(4, 4)] * 5 + [(5, 4)] * 5 + [(4, 4)] * 5  # straight back from the next well: no new event for (4, 4)
    assert [e for e in (d4.update(i / 10, w) for i, w in enumerate(seq4)) if e] == [(4, 4), (5, 4)]


def test_dwell_one_dispense_per_trip():
    from pipette_coach.tips import DwellDetector

    def run(seq, **kw):
        d, out = DwellDetector(min_s=0.4, gap_s=0.25, trip_gap_s=1.0, **kw), []
        for i, w in enumerate(seq):
            e = d.update(i / 10, w)
            if e is not None:
                out.append(("revise" if d.revised else "new", e))
        return out

    # paused 0.5 s next to the target on the way in, then 1.5 s on the target: the target replaces it
    assert run([(4, 4)] * 6 + [(5, 4)] * 15) == [("new", (4, 4)), ("revise", (5, 4))]
    # long stop, then a short pause on the way out: no change
    assert run([(5, 4)] * 15 + [(4, 4)] * 6) == [("new", (5, 4))]
    # two trips (tip off the plate 2 s in between): two dispenses
    assert run([(5, 4)] * 6 + [None] * 20 + [(5, 5)] * 6) == [("new", (5, 4)), ("new", (5, 5))]


def test_dwell_without_trip_rule_logs_each_well_of_one_pass():
    from pipette_coach.tips import DwellDetector

    d, out = DwellDetector(min_s=0.4, gap_s=0.25, trip_gap_s=None), []
    for i in range(90):  # tip never leaves the plate: 1 s each on D2, D3, D4
        out.append(d.update(i / 30, [(1, 3), (2, 3), (3, 3)][i // 30]))
    assert [e for e in out if e is not None] == [(1, 3), (2, 3), (3, 3)]


def test_dwell_two_seconds_is_a_dispense():
    """live_demo's default (no liquid model): 2 s on one well = dispensed; a short tip dropout does not restart."""
    from pipette_coach.tips import DwellDetector

    d = DwellDetector(min_s=2.0, gap_s=0.5, trip_gap_s=None)
    t = [k / 30 for k in range(90)]  # 3 s at 30 fps on A1, the tip missed between 1.0 and 1.3 s
    out = [(round(x, 2), d.update(x, None if 1.0 <= x < 1.3 else (0, 0))) for x in t]
    fired = [(x, w) for x, w in out if w is not None]
    assert fired == [(2.0, (0, 0))]  # once, at the 2 s mark
    d2 = DwellDetector(min_s=2.0, gap_s=0.5, trip_gap_s=None)
    assert all(d2.update(k / 30, (1, 1)) is None for k in range(57))  # 1.9 s of hovering: not a dispense
