"""Start and stop the live camera (scripts/live_demo.py --ui) from the page.

The camera code runs as its own process, so a crash or a slow model never takes the web server down. It opens its
own window with the camera picture and reports dispenses back to this server exactly as when started by hand.
Its output goes to data/liveCamera.log; the last lines are shown on the page if it stops by itself.
"""

import base64
import json
import os
import subprocess
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LIVE_DEMO = os.path.join(REPO, "scripts", "live_demo.py")


def liveCommand(camera, uiUrl, show=True):
    """The command that runs the live camera and reports to this server."""
    cmd = [sys.executable, LIVE_DEMO, "--camera", str(camera), "--ui", uiUrl]
    return cmd + ["--show"] if show else cmd


class LiveCamera:
    """One live camera process at a time. makeCommand(camera, show) -> argv (tests swap in a stand-in)."""

    def __init__(self, makeCommand, logPath):
        self.makeCommand = makeCommand
        self.logPath = logPath
        self.proc = None
        self.camera = None
        self.startedAt = None
        self.logFile = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, camera="0", show=True):
        camera = str(camera).strip() or "0"
        if self.running():
            raise ValueError("The live camera is already running. Stop it first.")
        self.closeLog()
        os.makedirs(os.path.dirname(self.logPath), exist_ok=True)
        self.logFile = open(self.logPath, "w")  # noqa: SIM115 (kept open while the process writes to it)
        self.proc = subprocess.Popen(
            self.makeCommand(camera, show), cwd=REPO, stdout=self.logFile, stderr=subprocess.STDOUT
        )
        self.camera = camera
        self.startedAt = time.time()

    def stop(self, wait=5.0):
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(wait)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(wait)
        self.closeLog()

    def closeLog(self):
        if self.logFile:
            self.logFile.close()
            self.logFile = None

    def logTail(self, n=6):
        try:
            with open(self.logPath, errors="replace") as f:
                return [ln.rstrip() for ln in deque(f, n) if ln.strip()]
        except OSError:
            return []

    def snapshot(self):
        """For the page: running or not, which camera, and why it stopped if it stopped by itself."""
        if self.proc is None:
            return {"running": False, "camera": None, "exitCode": None, "log": []}
        code = self.proc.poll()
        return {"running": code is None, "camera": self.camera, "exitCode": code,
                "log": [] if code in (None, 0, -15) else self.logTail()}  # fmt: skip


def cameraNames():
    """Names of the cameras macOS knows (e.g. "FaceTime HD Camera"), in the system's order; [] elsewhere or on error."""
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True,
                             timeout=10).stdout  # fmt: skip
        return [c.get("_name", "") for c in json.loads(out).get("SPCameraDataType", [])]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def openCamera(index):
    """cv2.VideoCapture for camera `index` (AVFoundation on a Mac), or None if it does not open."""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def findCameras(maxIndex=6, opener=openCamera):
    """Every camera that opens and gives a picture: [{"index", "width", "height", "thumb" (JPEG data URL), "name"}].
    Stops after two indexes in a row fail. Names come from macOS in its order: matched only when the counts agree."""
    import cv2

    found, misses = [], 0
    for i in range(maxIndex):
        cap = opener(i)
        frame = None
        if cap is not None:
            for _ in range(10):  # the first frames can be black while the camera wakes up
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
                    if f.mean() > 8:
                        break
            cap.release()
        if frame is None:
            misses += 1
            if misses >= 2:
                break
            continue
        misses = 0
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (240, max(1, round(240 * h / w))))
        jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        found.append({"index": i, "width": w, "height": h, "name": "",
                      "thumb": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()})  # fmt: skip
    names = cameraNames()
    if len(names) == len(found):
        for c, n in zip(found, names, strict=True):
            c["name"] = n
    return found
