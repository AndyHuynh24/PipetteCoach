"""Simulated use of scripts/label_plate_corners.py (no window)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import label_plate_corners as lpc  # noqa: E402


def make_tool(tmp_path):
    frames = [(0, np.zeros((720, 1280, 3), np.uint8)), (300, np.zeros((720, 1280, 3), np.uint8))]
    return lpc.CornerTool(frames, 30.0, tmp_path / "c.json", train_until=5.0)


def test_four_clicks_then_enter_saves_ordered_corners_and_advances(tmp_path):
    tool = make_tool(tmp_path)
    for x, y in ((900, 200), (300, 210), (310, 600), (890, 610)):  # any order
        tool.click(x, y)
    tool.render()
    assert tool.k == 0 and not (tmp_path / "c.json").exists()  # nothing saved before Enter
    tool.key(13)
    lab = json.loads((tmp_path / "c.json").read_text())["labels"]["0"]
    c = np.array(lab["corners"])
    assert c.shape == (4, 2) and tool.k == 1
    ang = np.arctan2(c[:, 1] - c[:, 1].mean(), c[:, 0] - c[:, 0].mean())
    assert np.all(np.diff(ang) > 0)  # sorted around the centre


def test_incomplete_corners_do_not_save_and_click_moves_nearest(tmp_path):
    tool = make_tool(tmp_path)
    tool.click(100, 100)
    tool.click(500, 100)
    tool.key(13)
    assert tool.k == 0 and "2/4" in tool.msg
    tool.click(505, 102)  # near the second corner: moves it, does not add a third
    assert len(tool.pts) == 2 and np.allclose(tool.pts[1], np.array([505, 102]) / tool.s)
    tool.key(ord("u"))
    assert len(tool.pts) == 1


def test_no_plate_saves_empty(tmp_path):
    tool = make_tool(tmp_path)
    tool.key(ord("x"))
    assert json.loads((tmp_path / "c.json").read_text())["labels"]["0"]["corners"] == []
