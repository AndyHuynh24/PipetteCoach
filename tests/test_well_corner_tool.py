"""Simulated use of scripts/label_well_corners.py (no window)."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import label_well_corners as lwc  # noqa: E402

from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.localizer.geometry import project  # noqa: E402

CLICKS = [(400, 200), (800, 205), (805, 470), (395, 465)]  # A1, A12, H12, H1 in view px (= original px here)
H_TR = cv2.findHomography(lwc.CORNERS, np.array(CLICKS, np.float32), 0)[0]
BOX = np.array([370.0, 175.0, 830.0, 495.0])


def make_tool(tmp_path, n=4, proposals=None):
    frame = np.zeros((720, 1280, 3), np.uint8)  # blank: the opening snap finds nothing and keeps the corner fit
    return lwc.WellCornerTool(list(range(0, 48 * n, 48)), lambda i: frame, 24.0, tmp_path / "w.json",
                              load_config("localizer"), (1280, 720), proposals)  # fmt: skip


def labels(tmp_path):
    return json.loads((tmp_path / "w.json").read_text())["labels"]


def drag(tool, a, b):
    tool.on_mouse(cv2.EVENT_LBUTTONDOWN, *a, 0, None)
    tool.on_mouse(cv2.EVENT_MOUSEMOVE, *b, 0, None)
    tool.on_mouse(cv2.EVENT_LBUTTONUP, *b, 0, None)


def test_tracker_grid_is_proposed_and_enter_keeps_it(tmp_path):
    tool = make_tool(tmp_path, proposals={i: (H_TR, BOX) for i in range(0, 192, 48)})
    assert tool.source == "tracker" and np.allclose(tool.corners, CLICKS)
    tool.render()
    tool.key(13)
    e = labels(tmp_path)["0"]
    assert tool.k == 1 and e["source"] == "tracker" and np.allclose(e["H"], H_TR)


def test_dragging_a_corner_refits_and_is_carried_while_the_box_is_still(tmp_path):
    moved = BOX + [0, 0, 0, 40]  # frame 96: the plate box moved
    tool = make_tool(tmp_path, proposals={0: (H_TR, BOX), 48: (H_TR, BOX + 2), 96: (H_TR, moved)})
    drag(tool, (402, 198), (410, 190))  # grab near A1, put it 8 px right / 10 up
    assert tool.source == "adjusted" and tool.drag is None
    assert np.allclose(project(tool.H, lwc.CORNERS[:1])[0], (410, 190), atol=0.01)
    assert np.allclose(project(tool.H, lwc.CORNERS[1:]), CLICKS[1:], atol=0.01)  # other corners stay
    H_fixed = tool.H
    tool.key(13)
    assert tool.source == "carried" and np.allclose(tool.H, H_fixed)  # box still: your fixed grid, not the tracker's
    tool.key(ord("t"))
    assert tool.source == "tracker"
    tool.key(ord("t"))
    tool.key(13)
    assert tool.source == "tracker"  # box moved: back to the tracker's grid
    assert labels(tmp_path)["48"]["source"] == "carried"


def test_no_proposal_click_from_scratch_and_x_splits_stretches(tmp_path):
    tool = make_tool(tmp_path, n=5)
    tool.key(13)
    assert tool.k == 0 and "0/4" in tool.msg  # nothing to save yet
    for x, y in CLICKS:
        tool.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        tool.on_mouse(cv2.EVENT_LBUTTONUP, x, y, 0, None)
    assert tool.source == "clicked" and np.allclose(project(tool.H, lwc.CORNERS), CLICKS, atol=0.01)
    tool.key(13)  # 0
    tool.key(13)  # 48 carried
    tool.key(ord("x"))  # 96 can't tell
    assert tool.H is None
    for x, y in CLICKS:
        tool.press(x + 30, y)
    tool.key(13)  # 144 new grid
    tool.key(13)  # 192 carried
    st = lwc.stretches(labels(tmp_path))
    assert [(s["f0"], s["f1"]) for s in st] == [(0, 48), (144, 192)]


def test_clicks_when_zoomed_land_in_original_pixels(tmp_path):
    tool = make_tool(tmp_path)
    tool.zoom(640, 360)  # 3x around the centre of the view
    assert np.isclose(tool.s, 3.0)
    tool.press(0, 0)
    assert np.allclose(tool.corners[0], tool.view[:2])  # top-left of the zoomed view
    assert np.allclose(tool.to_view(tool.corners[0]), 0)
    tool.render()
    tool.zoom(0, 0)
    assert tool.view == (0.0, 0.0, 1280.0) and np.isclose(tool.s, 1.0)


def test_flip_turns_the_labelling_180_degrees(tmp_path):
    tool = make_tool(tmp_path, proposals={0: (H_TR, BOX)})
    tool.key(ord("f"))
    assert tool.source == "adjusted"
    assert np.allclose(project(tool.H, lwc.CORNERS), [CLICKS[2], CLICKS[3], CLICKS[0], CLICKS[1]], atol=0.01)


def test_other_view_of_the_same_clip_keeps_the_grid(tmp_path):
    seen = []
    frame = np.zeros((720, 1280, 3), np.uint8)
    tool = lwc.WellCornerTool([0, 1000], lambda c: seen.append(c) or frame, 1000.0, tmp_path / "w.json",
                              load_config("localizer"), (1280, 720), {0: (H_TR, None), 1000: (H_TR, None)})  # fmt: skip
    tool.n_alt = 5
    tool.lab["clips"] = ["clip_a", "clip_b"]
    drag(tool, (402, 198), (410, 190))
    H = tool.H
    tool.key(ord("n"))
    tool.key(ord("n"))
    assert seen[-1] == 2 and tool.alt == 2 and np.allclose(tool.H, H) and tool.source == "adjusted"
    tool.render()
    tool.key(13)
    assert labels(tmp_path)["0"]["source"] == "adjusted" and tool.alt == 0 and seen[-1] == 1000
