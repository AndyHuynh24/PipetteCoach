"""Auto-label the plate outline on a video with the wells-first finder (localizer/wells_first.py), per chunk.

Only chunks where all 12 x 8 wells are visible get a label (outline from the lattice + measured rim, edges where
they agree); others are skipped. Output: data/processed/video_labels/<name>.json {fps, chunks: [{start, end,
outline (original px), scale}]} + a contact sheet runs/<ts>_video-autolabel/sheet.jpg to check by eye.

    python scripts/autolabel_video_plate.py --video data/raw/pipette1.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.localizer.wells_first import find_plate  # noqa: E402
from pipette_coach.runs import new_run_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--chunk", type=float, default=3.0)
    a = ap.parse_args()
    cfg = load_config("localizer")
    det = PlateDetector(cfg)
    net = WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_chunk = int(a.chunk * fps)
    chunks, thumbs, i0, t0 = [], [], 0, time.time()
    while True:
        frames = []
        for _ in range(n_chunk):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        if len(frames) < n_chunk // 2:
            break
        r = find_plate(frames, det, net, cfg)
        if r is not None and r["full"] and r["outline"] is not None:
            chunks.append({"start": i0, "end": i0 + len(frames), "outline": r["outline"].round(2).tolist(),
                           "scale": r["scale"]})  # fmt: skip
            img = frames[len(frames) // 2].copy()
            cv2.polylines(img, [r["outline"].astype(np.int32)], True, (0, 0, 255), 3)
            c = r["outline"].mean(0)
            x0, y0 = np.maximum((c - 300).astype(int), 0)
            th = cv2.resize(img[y0 : y0 + 600, x0 : x0 + 600], (240, 240))
            cv2.putText(th, f"{i0 / fps:.0f}s", (4, 16), 0, 0.5, (0, 255, 255), 1)
            thumbs.append(th)
        i0 += len(frames)
        print(f"{i0 / fps:6.1f}s: {'LABEL' if chunks and chunks[-1]['start'] == i0 - len(frames) else 'skip '} "
              f"({len(chunks)} labelled, {time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    out = PROCESSED_DIR / "video_labels"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(a.video).stem}.json").write_text(json.dumps({"video": a.video, "fps": fps, "chunks": chunks}))
    run = new_run_dir("video-autolabel")
    if thumbs:
        rows = [np.hstack(thumbs[k : k + 8] + [np.zeros_like(thumbs[0])] * (8 - len(thumbs[k : k + 8])))
                for k in range(0, len(thumbs), 8)]  # fmt: skip
        cv2.imwrite(str(run / "sheet.jpg"), np.vstack(rows))
    note = f"{len(chunks)} chunks of {a.chunk}s labelled in {a.video}. Check sheet.jpg by eye. Keep local."
    (run / "notes.md").write_text(f"# Auto-labelled plate outlines\n\n{note}\n")
    print(f"{len(chunks)} chunks labelled -> {out}; sheet {run / 'sheet.jpg'}")


if __name__ == "__main__":
    main()
