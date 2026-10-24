"""Which camera index is which (OpenCV gives numbers, not names): grabs one frame from each index that opens, prints
its resolution and saves it as runs/cameras/cam<i>.jpg so you can see which one is the phone.

    python scripts/list_cameras.py            # indexes 0-5
    python scripts/list_cameras.py --max 3
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=5, help="highest index to try")
    ap.add_argument("--size", default="1920x1080", help="resolution to ask for (WxH)")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    out = REPO_ROOT / "runs" / "cameras"
    out.mkdir(parents=True, exist_ok=True)
    found = 0
    for i in range(a.max + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        frame, t0 = None, time.time()
        while time.time() - t0 < 3:  # the first frames can be black while the camera starts
            ok, f = cap.read()
            if ok and f is not None and f.mean() > 5:
                frame = f
                break
        size = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        if frame is not None:  # what it really sends (a camera can report the asked size and send another)
            size = f"{frame.shape[1]}x{frame.shape[0]}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        found += 1
        if frame is None:
            print(f"camera {i}: {size} at {fps:.0f} fps, no picture (covered, asleep, or no permission)")
            continue
        path = out / f"cam{i}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"camera {i}: {size} at {fps:.0f} fps -> {path}")
    if not found:
        print("No camera opened. On a Mac: System Settings > Privacy & Security > Camera, allow your terminal app.")


if __name__ == "__main__":
    main()
