"""Draw the "How it works" layers on a raw run recording, one clip per layer, from the per-frame detections that
`scripts/live_demo.py --dump` writes. Run from the repo root:
    python Videos/demo-video/make_layers.py <run video.mp4> <dump.jsonl>
Writes Videos/demo-video/public/layers/<layer>.mp4 (H.264, 1920x1080, 30 fps)."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipette_coach.localizer.geometry import GRID, OUTLINE, pitch_px, project  # noqa: E402

TEAL, GREEN, AMBER, WHITE = (160, 200, 40), (90, 200, 60), (40, 170, 255), (255, 255, 255)  # BGR
# layer: (first s, last s) of the run video
LAYERS = {"camera": (20, 26), "plate": (20, 26), "wells": (20, 26), "tip": (22, 28), "dwell": (23.5, 29.5)}
ORDER = list(LAYERS)


def frames(video, a, b):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000)
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, f = cap.read()
        if not ok or t > b:
            break
        yield t, f


def nearest(rows, t):
    i = min(range(len(rows)), key=lambda k: abs(rows[k]["t"] - t))
    return rows[i]


def draw(img, layer, r, trail, k, last_H, done):
    upto = ORDER.index(layer)
    H = r["H"] or last_H
    if H is None:
        return img
    H = np.asarray(H)
    over = img.copy()
    if upto >= 1:  # the plate: outline, drawn in over the first second
        o = project(H, OUTLINE).astype(np.int32)
        if layer == "plate":
            cv2.fillPoly(over, [o], TEAL)
            img = cv2.addWeighted(over, 0.18 * min(k / 20, 1), img, 1 - 0.18 * min(k / 20, 1), 0)
        cv2.polylines(img, [o], True, TEAL, 6, cv2.LINE_AA)
    if upto >= 2:  # all 96 wells, appearing column by column
        c = project(H, GRID)
        rad = max(4, int(pitch_px(H) * 0.36))
        show = 96 if layer != "wells" else int(min(k / 45, 1) * 12 + 0.999) * 8
        for j, p in enumerate(c):
            if (j % 12) * 8 + j // 12 < show:
                cv2.circle(img, tuple(np.round(p).astype(int)), rad, GREEN, 3, cv2.LINE_AA)
    if upto >= 3 and r["tip"] is not None:  # the tip, its trail, and the well under it
        pts = np.array(trail, np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, AMBER, 4, cv2.LINE_AA)
        tp = tuple(np.round(r["tip"]).astype(int))
        cv2.drawMarker(img, tp, AMBER, cv2.MARKER_CROSS, 44, 5)
        cv2.circle(img, tp, 18, AMBER, 3, cv2.LINE_AA)
        if r["well"]:
            c = project(H, GRID)
            w = c[np.argmin(np.linalg.norm(c - np.array(r["tip"]), axis=1))]
            rad = int(pitch_px(H) * 0.6)
            wc = tuple(np.round(w).astype(int))
            if layer == "dwell":  # the hold timer fills up; full = dispensed
                prog = r["dwell"]
                col = GREEN if prog >= 1 else AMBER
                cv2.ellipse(img, wc, (rad + 6, rad + 6), -90, 0, 360 * prog, col, 9, cv2.LINE_AA)
            else:
                cv2.circle(img, wc, rad, AMBER, 4, cv2.LINE_AA)
            cv2.putText(img, r["well"], (wc[0] - rad - 70, wc[1] - rad - 10), cv2.FONT_HERSHEY_DUPLEX, 1.6, WHITE, 6,
                        cv2.LINE_AA)  # fmt: skip
            cv2.putText(img, r["well"], (wc[0] - rad - 70, wc[1] - rad - 10), cv2.FONT_HERSHEY_DUPLEX, 1.6, AMBER, 3,
                        cv2.LINE_AA)  # fmt: skip
    if layer == "dwell" and done:
        cv2.putText(img, f"{done} dispensed", (60, 1020), cv2.FONT_HERSHEY_DUPLEX, 2.2, WHITE, 10, cv2.LINE_AA)
        cv2.putText(img, f"{done} dispensed", (60, 1020), cv2.FONT_HERSHEY_DUPLEX, 2.2, GREEN, 4, cv2.LINE_AA)
    return img


def main(video, dump):
    rows = [json.loads(x) for x in open(dump)]
    out_dir = Path(__file__).parent / "public" / "layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer, (a, b) in LAYERS.items():
        out = out_dir / f"{layer}.mp4"
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
                                "-r", "30", "-i", "-", "-c:v", "h264_videotoolbox", "-b:v", "10M", "-pix_fmt", "yuv420p",
                                str(out)], stdin=subprocess.PIPE)  # fmt: skip
        trail, last_H, done = [], None, None
        for k, (t, f) in enumerate(frames(video, a, b)):
            r = nearest(rows, t)
            last_H = r["H"] or last_H
            if r["tip"] is not None:
                trail = (trail + [np.round(r["tip"]).astype(int).tolist()])[-14:]
            if r["dwell"] >= 1 and r["well"]:
                done = r["well"]
            f = cv2.resize(f, (1920, 1080))
            enc.stdin.write(draw(f, layer, r, trail, k, last_H, done).tobytes())
        enc.stdin.close()
        enc.wait()
        print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
