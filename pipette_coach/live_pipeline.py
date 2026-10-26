"""Real-time pipeline for the live demo: camera -> plate grid -> tip -> well -> dispense, at camera rate.

Three parts, so the slow, rare plate steps (detector, rebuild, periodic check) never stall the video:
  LatestFrame  camera thread: always holds the NEWEST frame (no backlog -> no growing lag). For a video file it
               plays at the file's frame rate, dropping frames the consumer did not take (as a camera would).
  PlateWorker  thread: PlateTracker on the newest frame, as often as it can (~15-25 Hz); the plate sits still,
               so a grid a few frames old is fine. Its result is read by the main loop.
  TipFinder    main loop, every camera frame: tip model on the plate crop -> tip end -> well (tips.well_of);
               the caller feeds the well to tips.DwellDetector.
"""

import threading
import time

import cv2
import numpy as np

from pipette_coach.tips import plate_crop_box, well_of

GPU_LOCK = threading.Lock()


def serialize_gpu(det, net, tip_model) -> None:
    """The Apple GPU backend (MPS) crashes when two threads encode work at once ("A command encoder is already
    encoding to this command buffer"). Every model call takes one shared lock for the duration of ONE call (results
    copied to the CPU inside it), so the threads take turns per call: the main loop waits at most one plate-model
    call (~10-30 ms), never a whole rebuild, and the tracker's CPU work still overlaps with the tip model."""

    def yolo(model):
        predict = model.predict

        def locked(*a, **k):
            with GPU_LOCK:
                return [r.cpu() for r in predict(*a, **k)]

        model.predict = locked

    yolo(det.model)
    yolo(tip_model)
    heatmap = net.heatmap

    def locked_heatmap(*a, **k):
        with GPU_LOCK:
            return heatmap(*a, **k)

    net.heatmap = locked_heatmap


def is_stream(src) -> bool:
    """A live network camera (e.g. a phone app's http://192.168.1.20:8080/video), not a file."""
    return isinstance(src, str) and src.lower().startswith(("http://", "https://", "rtsp://"))


class LatestFrame:
    """Background reader that keeps only the newest frame. src: camera index, stream URL or video path. size: (w, h)
    to ask a camera for (it may give something else; see self.size)."""

    def __init__(self, src, realtime: bool = True, start_s: float = 0.0, size=None):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {src}")
        if size and not isinstance(src, str):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.is_file = isinstance(src, str) and not is_stream(src)
        if start_s and self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * (self.cap.get(cv2.CAP_PROP_FPS) or 30.0)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime = realtime
        self.flip: int | None = None  # cv2.flip code for every frame (1 left-right, 0 up-down, -1 both); None = as is
        self.frame, self.idx, self.t = None, -1, 0.0
        self.read_count = 0
        self.done = False
        self.end_reason = None
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        t0 = time.perf_counter()
        i = 0
        while not self.done:
            ok, f = self.cap.read()
            if not ok:
                self.end_reason = f"no frame from the source after {i} frames"
                break
            if self.is_file and self.realtime:  # play the file at its own rate, like a camera
                wait = t0 + i / self.fps - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            if self.flip is not None:  # camera mounted mirrored / upside down: every user of the frame sees it fixed
                f = cv2.flip(f, self.flip)
            if i == 0:
                self.size = (f.shape[1], f.shape[0])  # what the camera really sends (it may ignore the asked size)
            with self.cond:
                self.frame, self.idx, self.t = f, i, (i / self.fps if self.is_file else time.perf_counter() - t0)
                self.read_count += 1
                self.cond.notify_all()
            i += 1
        with self.cond:
            self.done = True
            self.cond.notify_all()

    def next(self, after: int, timeout: float = 2.0):
        """Newest frame with index > after: (idx, t, frame), or None when the source has ended."""
        with self.cond:
            self.cond.wait_for(lambda: self.idx > after or self.done, timeout)
            if self.idx > after:
                return self.idx, self.t, self.frame
            return None

    def stop(self) -> None:
        self.done = True
        self.thread.join(timeout=2)
        self.cap.release()


class PlateWorker:
    """PlateTracker on the newest frame in a thread; latest() returns its last result (dict as from update)."""

    def __init__(self, tracker, cam: LatestFrame):
        self.tracker, self.cam = tracker, cam
        self.out: dict = {"status": "starting", "H": None, "box": None}
        self.updates, self.stop_flag = 0, False
        self.log: list[tuple[int, str]] = []  # (frame index, tracker status) per update, for checking a run
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        last = -1
        while not self.stop_flag:
            got = self.cam.next(last, timeout=0.5)
            if got is None:
                if self.cam.done:
                    break
                continue
            last, _, frame = got
            out = self.tracker.update(frame)
            with self.lock:
                self.out = {**out, "frame_idx": last}
                self.updates += 1
                self.log.append((last, out["status"]))

    def latest(self) -> dict:
        with self.lock:
            return self.out

    def stop(self) -> None:
        self.stop_flag = True
        self.thread.join(timeout=5)


class TipFinder:
    """Most confident tip end on the plate crop (outline + margin) -> (tip xy in the frame or None, well or None)."""

    def __init__(self, model, conf: float = 0.3, imgsz: int = 480, margin: float = 0.25, device=None):
        self.model, self.conf, self.imgsz, self.margin, self.device = model, conf, imgsz, margin, device

    def __call__(self, frame: np.ndarray, plate: dict):
        if plate.get("outline") is None or plate.get("H") is None:
            return None, None
        x0, y0, x1, y1 = plate_crop_box(np.asarray(plate["outline"]), self.margin, frame.shape[1], frame.shape[0])
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None, None
        r = self.model.predict(frame[y0:y1, x0:x1], imgsz=self.imgsz, conf=self.conf, verbose=False,
                               device=self.device)[0]  # fmt: skip
        if not len(r.boxes):
            return None, None
        k = int(r.boxes.conf.argmax())
        tip = r.keypoints.xy.cpu().numpy()[k, 0] + [x0, y0]
        return tip, well_of(plate["H"], tip)
