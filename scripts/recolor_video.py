"""Make simulated-colour copies of a video with a dyed liquid (pipette_coach/recolor.py), e.g. red -> blue / clear.

Same frames, size and fps as the input, so the input's CVAT tip labels (frame ids) apply to every copy. Output:
data/processed/recolor/<video>_<color>.mp4, and with --preview a half-size side-by-side (original | recoloured)
<video>_<color>_compare.mp4 to check by eye.

    python scripts/recolor_video.py --video data/raw/pipette6.mov --colors blue --preview
    python scripts/recolor_video.py --video data/raw/pipette6.mov --colors blue,green,yellow,purple,orange,clear
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.recolor import COLORS, dye_weight, recolor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--colors", default="blue", help=f"comma-separated, from {', '.join(COLORS)}")
    ap.add_argument("--preview", action="store_true", help="also write a side-by-side comparison video")
    ap.add_argument("--out", default=str(PROCESSED_DIR / "recolor"))
    a = ap.parse_args()
    colors = a.colors.split(",")
    for c in colors:
        if c not in COLORS:
            sys.exit(f"unknown colour {c}; choose from {', '.join(COLORS)}")
    cap = cv2.VideoCapture(a.video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem, four = Path(a.video).stem, cv2.VideoWriter_fourcc(*"mp4v")
    writers = {c: cv2.VideoWriter(str(out / f"{stem}_{c}.mp4"), four, fps, (w, h)) for c in colors}
    previews = {c: cv2.VideoWriter(str(out / f"{stem}_{c}_compare.mp4"), four, fps, (w, h // 2)) for c in colors
                if a.preview}  # fmt: skip
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        wt = dye_weight(frame)
        for c in colors:
            new = recolor(frame, wt, c)
            writers[c].write(new)
            if c in previews:
                pair = np.hstack([frame, new])
                pair = cv2.resize(pair, (w, h // 2), interpolation=cv2.INTER_AREA)
                cv2.putText(pair, "original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(pair, c, (w // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                previews[c].write(pair)
        if i % 500 == 0:
            print(f"{i}/{n}", flush=True)
    for wr in [*writers.values(), *previews.values()]:
        wr.release()
    for c in colors:
        print(
            out / f"{stem}_{c}.mp4",
            "| compare:" if c in previews else "",
            out / f"{stem}_{c}_compare.mp4" if c in previews else "",
        )


if __name__ == "__main__":
    main()
