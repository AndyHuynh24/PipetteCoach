import cv2
import numpy as np

from pipette_coach.recorder import VideoRecorder


def test_recorder_plays_in_real_time(tmp_path):
    rec = VideoRecorder(tmp_path / "run.mp4", fps=10)
    img = np.zeros((48, 64, 3), np.uint8)
    for k in range(10):  # 10 frames over 2 s (5 fps loop): each is written twice to fill 10 fps
        img[:] = k * 20
        rec.write(img.copy(), k * 0.2)
    rec.write(img.copy(), 1.85)  # same 0.1 s slot as the frame before: skipped, not written
    rec.close()
    assert (rec.frames, rec.dropped) == (19, 0)
    cap = cv2.VideoCapture(str(tmp_path / "run.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 19
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (64, 48)


def test_recorder_without_frames_writes_nothing(tmp_path):
    rec = VideoRecorder(tmp_path / "empty.mp4")
    rec.close()
    assert not (tmp_path / "empty.mp4").exists()
