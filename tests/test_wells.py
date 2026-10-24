import pytest

from pipette_coach.protocol.wells import all_wells, nearest_well, parse_well, well, well_center, well_str


def test_roundtrip_all_wells():
    ws = all_wells()
    assert len(ws) == 96 and ws[0] == {"row": "A", "col": 1} and ws[-1] == {"row": "H", "col": 12}
    for w in ws:
        assert nearest_well(*well_center(w)) == w
        assert parse_well(well_str(w)) == w


def test_canonical_convention():
    assert well_center(well("A", 1)) == (0.0, 0.0)
    assert well_center(well("H", 12)) == (11.0, 7.0)
    assert well_center(well("B", 7)) == (6.0, 1.0)


def test_nearest_well_rejects_off_plate_and_between_wells():
    assert nearest_well(-1.0, 0.0) is None
    assert nearest_well(12.0, 3.0) is None
    assert nearest_well(0.5, 0.5) is None  # equidistant from 4 wells: not a confident well
    assert nearest_well(3.2, 4.1) == {"row": "E", "col": 4}


@pytest.mark.parametrize("bad", [("I", 1), ("A", 0), ("A", 13)])
def test_invalid_wells(bad):
    with pytest.raises(ValueError):
        well(*bad)
