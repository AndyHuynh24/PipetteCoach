"""Record a video (e.g. the raw camera frames of a run) to an mp4 without slowing the camera loop.

write() only puts the frame on a queue; a background thread encodes. The video plays in real time: each frame is
written for as many 1/fps slots as passed since the last one (the loop may run slower than fps), and a full queue
drops frames instead of blocking (counted in `dropped`).
"""

import queue
import threading
from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, path: Path, fps: float = 30.0, max_queue: int = 90, max_repeat: int = 10):
        self.path, self.fps, self.max_repeat = Path(path), fps, max_repeat
        self.frames, self.dropped = 0, 0  # written to the file / thrown away (queue full)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._t0: float | None = None
        self._slots = 0  # 1/fps slots handed out so far
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, img: np.ndarray, t: float) -> None:
        """Frame shown at time t (seconds, any clock that only goes forward). Returns at once."""
        if self._t0 is None:
            self._t0 = t
        due = int((t - self._t0) * self.fps) + 1  # slots that should be filled by now
        n = min(max(due - self._slots, 0), self.max_repeat)
        if n == 0:
            return  # faster than fps: this frame's slot is already taken
        try:
            self._q.put_nowait((img, n))
            self._slots = due
        except queue.Full:
            self.dropped += 1

    def close(self, wait_s: float = 10.0) -> None:
        """Write what is queued, then finish the file."""
        self._q.put(None)
        self._thread.join(timeout=wait_s)

    def _run(self) -> None:
        vw = None
        while True:
            item = self._q.get()
            if item is None:
                break
            img, n = item
            if vw is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, img.shape[1::-1])
            for _ in range(n):
                vw.write(img)
                self.frames += 1
        if vw is not None:
            vw.release()
