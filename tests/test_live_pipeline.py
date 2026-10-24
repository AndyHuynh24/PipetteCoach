import time

import cv2
import numpy as np

from pipette_coach.live_pipeline import LatestFrame, PlateWorker, is_stream


def _video(path, n=30, fps=30.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        vw.write(np.full((48, 64, 3), i * 8 % 256, np.uint8))
    vw.release()
    return str(path)


def test_latest_frame_fast_consumer_gets_every_frame(tmp_path):
    cam = LatestFrame(_video(tmp_path / "a.mp4"), realtime=True)
    seen, last = [], -1
    while (got := cam.next(last)) is not None:
        last = got[0]
        seen.append(last)
    cam.stop()
    assert seen[-1] == 29 and len(seen) >= 27  # real-time pacing: a fast consumer keeps up


def test_latest_frame_slow_consumer_gets_newest(tmp_path):
    cam = LatestFrame(_video(tmp_path / "b.mp4", n=60), realtime=True)
    seen, last = [], -1
    while (got := cam.next(last)) is not None:
        last = got[0]
        seen.append(last)
        time.sleep(0.1)  # 10 Hz consumer of a 30 fps source: frames are skipped, never queued
    cam.stop()
    assert len(seen) < 30 and seen[-1] >= 55
    assert all(b > a for a, b in zip(seen, seen[1:], strict=False))


def test_latest_frame_start_offset(tmp_path):
    cam = LatestFrame(_video(tmp_path / "c.mp4", n=60), realtime=False, start_s=1.0)
    got = cam.next(-1)
    cam.stop()
    assert got is not None and abs(int(got[2][0, 0, 0]) - 30 * 8) <= 6  # first frame read is frame 30 (mp4 lossy)


class _FakeTracker:
    def __init__(self):
        self.calls = 0

    def update(self, frame):
        self.calls += 1
        time.sleep(0.05)  # slow tracker (20 Hz) on a 30 fps source
        return {"status": "tracked", "H": np.eye(3), "box": None, "value": int(frame[0, 0, 0])}


def test_plate_worker_runs_on_newest_frames(tmp_path):
    cam = LatestFrame(_video(tmp_path / "d.mp4", n=45), realtime=True)
    tr = _FakeTracker()
    w = PlateWorker(tr, cam)
    last = -1
    while (got := cam.next(last)) is not None:
        last = got[0]
    time.sleep(0.2)
    w.stop()
    cam.stop()
    assert 10 < tr.calls < 45  # did not process every frame, did not stall
    assert w.latest()["frame_idx"] >= 40


def test_latest_frame_size_is_only_asked_of_cameras(tmp_path):
    cam = LatestFrame(_video(tmp_path / "e.mp4"), realtime=False, size=(1920, 1080))
    assert cam.size == (64, 48)  # a file keeps its own size
    cam.stop()


def test_stream_urls_are_live_not_files():
    assert is_stream("http://192.168.1.20:8080/video") and is_stream("rtsp://10.0.0.5/live")
    assert not is_stream(0) and not is_stream("data/own/clip.mp4")
