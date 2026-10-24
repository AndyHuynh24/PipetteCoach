"""Record the plate from the camera and mark dispense moments by hand: hit SPACE the moment you inject.

Saves to data/own/<timestamp>/ :
    video.mp4         the raw camera frames (no overlay)
    frame_times.csv   frame index, recording seconds of each frame (pauses left out)
    dispenses.json    one entry per SPACE press: frame index + recording seconds

Keys (in the preview window): SPACE = mark a dispense, u = undo the last mark (press again to undo more),
p = pause / resume (nothing is recorded while paused), q / Esc = stop.

    python scripts/record_dispenses.py --camera 0
    python scripts/record_dispenses.py --camera 1 --size 1280x720
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT  # noqa: E402


def save_marks(path: Path, meta: dict, marks: list[dict]) -> None:
    # Rewritten on every press so a crash never loses the labels.
    path.write_text(json.dumps({**meta, "dispenses": marks}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0", help="camera index (0, 1, ...) or a stream URL")
    ap.add_argument("--size", default="1920x1080", help="resolution to ask the camera for (WxH)")
    ap.add_argument("--fps", type=float, default=30, help="record at this fps; faster cameras get frames dropped")
    ap.add_argument("--out", default=None, help="output folder (default data/own/<timestamp>)")
    a = ap.parse_args()

    cam = int(a.camera) if a.camera.isdigit() else a.camera
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {a.camera}. Try scripts/list_cameras.py, and on a Mac allow camera access.")
    w, h = (int(v) for v in a.size.split("x"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Camera opened but sent no frame.")
    h, w = frame.shape[:2]  # what it really sends
    fps = a.fps

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.out) if a.out else REPO_ROOT / "data" / "own" / stamp
    out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    meta = {"camera": str(a.camera), "size": [w, h], "fps": fps, "started": stamp}
    marks: list[dict] = []
    marks_path = out / "dispenses.json"
    save_marks(marks_path, meta, marks)

    times_f = open(out / "frame_times.csv", "w", newline="")
    times = csv.writer(times_f)
    times.writerow(["frame", "t"])

    print(f"Recording {w}x{h} @ {fps:.0f} fps -> {out}")
    print("SPACE = mark dispense, u = undo, p = pause/resume, q/Esc = stop")
    win = "record (SPACE = dispense, u = undo, p = pause, q = stop)"
    scale = min(1.0, 960 / w)  # small preview keeps the loop fast; the mp4 stays full size
    # t is recording time: paused stretches are left out of both the mp4 and t, so t keeps matching the video.
    t0 = time.monotonic()
    paused_at, paused_total = None, 0.0
    idx, flash_until, first = 0, 0.0, True
    try:
        while True:
            if not first:
                ok, frame = cap.read()
                if not ok:
                    print("Camera stopped sending frames.")
                    break
            first = False
            now = time.monotonic()
            t = (paused_at if paused_at is not None else now) - t0 - paused_total
            # A 60 fps camera is cut down to `fps` by writing only once each frame slot is due.
            if paused_at is None and t >= (idx - 0.25) / fps:
                writer.write(frame)
                times.writerow([idx, f"{t:.4f}"])
                idx += 1

            view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
            if now < flash_until:
                cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 255, 0), 12)
            status = "PAUSED (p to resume)" if paused_at is not None else "REC"
            cv2.putText(
                view,
                f"{status} {t:6.1f}s  frame {idx - 1}  marks {len(marks)}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255) if paused_at is not None else (0, 0, 255),
                2,
            )
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if paused_at is not None:
                    print("  paused: not marked (press p to resume first)")
                    continue
                # The mark belongs to the frame just written, i.e. what was on screen when you pressed.
                marks.append({"frame": idx - 1, "t": round(t, 4)})
                save_marks(marks_path, meta, marks)
                flash_until = now + 0.3
                print(f"  dispense #{len(marks)} at {t:.2f}s (frame {idx - 1})")
            elif key == ord("u"):
                if marks:
                    m = marks.pop()
                    save_marks(marks_path, meta, marks)
                    print(f"  undid mark at {m['t']:.2f}s (frame {m['frame']}), {len(marks)} left")
                else:
                    print("  nothing to undo")
            elif key == ord("p"):
                if paused_at is None:
                    paused_at = now
                    print(f"  paused at {t:.2f}s")
                else:
                    paused_total += now - paused_at
                    paused_at = None
                    print(f"  resumed at {t:.2f}s")
            elif key in (ord("q"), 27):
                break
    finally:
        if paused_at is not None:
            paused_total += time.monotonic() - paused_at
        dur = time.monotonic() - t0 - paused_total
        cap.release()
        writer.release()
        times_f.close()
        cv2.destroyAllWindows()
        meta.update(frames=idx, seconds=round(dur, 2), measured_fps=round(idx / max(dur, 1e-6), 2))
        save_marks(marks_path, meta, marks)
        print(f"Saved {idx} frames ({dur:.1f}s recorded, {meta['measured_fps']} fps), {len(marks)} marks -> {out}")
        if abs(meta["measured_fps"] - fps) > 2:
            print(
                f"Note: camera delivered ~{meta['measured_fps']} fps but the mp4 says {fps:.0f}; "
                "use the frame index (or frame_times.csv), not video seconds, to line up marks."
            )


if __name__ == "__main__":
    main()
