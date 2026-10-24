"""Simulated use of scripts/label_tips.py (no window): clicks place a marker without saving or moving on;
Save & Next saves and advances; No tip; multichannel first + last tip; Play; Clear."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pipette_coach.tracking.tip_data import TIP_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(not (TIP_DIR / "Plate_9_clip_0005.npz").exists(), reason="needs local tip data")


class NoModel:
    def heatmap(self, view):
        return None


def make_tool(tmp_path, clip="Plate_9_clip_0005", multi=False):
    import label_tips as lt

    wells = {clip: list(range(8)) if multi else [13]}
    return lt, lt.LabelTool([(clip, 20), (clip, 30)], wells, tmp_path / "clicks.json", NoModel())


def center(rect):
    return (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2


def test_click_places_marker_and_stays_then_next_saves(tmp_path):
    lt, tool = make_tool(tmp_path)
    tool.click_at(300, lt.TOP + 200)  # plate-view panel
    assert tool.pending_kind == "tip" and len(tool.pending) == 1 and tool.frame == 20  # stays on the frame
    assert not (tmp_path / "clicks.json").exists()  # nothing saved yet
    tool.click_at(320, lt.TOP + 210)  # click again: moves the marker
    assert len(tool.pending) == 1 and np.allclose(tool.pending[0], [320 / lt.ZV, 210 / lt.ZV])
    tool.render()
    tool.click_at(*center(tool.btn_rects["next"]))  # Save & Next button
    lab = json.loads((tmp_path / "clicks.json").read_text())["labels"]
    e = lab["Plate_9_clip_0005"]["20"]
    assert e["kind"] == "tip" and np.allclose(e["tips_view"][0], [320 / lt.ZV, 210 / lt.ZV], atol=0.01)
    assert tool.frame == 30 and tool.pending_kind is None
    tool.click_at(*center(tool.btn_rects["prev"]))  # back: the saved marker is shown again
    assert tool.frame == 20 and tool.pending_kind == "tip"


def test_next_without_marker_does_not_save_or_move(tmp_path):
    _, tool = make_tool(tmp_path)
    tool.action("next")
    assert tool.frame == 20 and "Nothing to save" in tool.msg


def test_no_tip_play_clear_and_camera_panel_click(tmp_path):
    lt, tool = make_tool(tmp_path)
    tool.key(ord("x"))
    assert tool.pending_kind == "none"
    tool.key(13)
    assert json.loads((tmp_path / "clicks.json").read_text())["labels"]["Plate_9_clip_0005"]["20"]["kind"] == "none"
    tool.click_at(*center(tool.btn_rects["play"]))
    assert tool.play and tool.play[-1] == tool.frame == 30
    tool.render()  # renders while playing
    tool.click_at(lt.VW + 250, lt.TOP + 250)  # camera panel click -> marker in plate-view coords
    assert tool.pending_kind == "tip" and 0 <= tool.pending[0][0] < 224 and not tool.play
    tool.key(ord("c"))
    assert tool.pending_kind is None


def test_multichannel_needs_first_and_last_tip(tmp_path):
    lt, tool = make_tool(tmp_path, multi=True)
    tool.click_at(200, lt.TOP + 100)
    tool.action("next")
    assert tool.frame == 20 and "LAST tip" in tool.msg or "1/2" in tool.msg
    tool.click_at(200, lt.TOP + 400)
    tool.action("next")
    e = json.loads((tmp_path / "clicks.json").read_text())["labels"]["Plate_9_clip_0005"]["20"]
    assert len(e["tips_view"]) == 8 and tool.frame == 30
