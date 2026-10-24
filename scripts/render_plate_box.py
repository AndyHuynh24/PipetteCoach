"""Render the plate bounding box measured from the plate's own edges (no grid homography involved).

Per clip: detector box as a rough seed -> per-frame edge rectangle (localizer/plate_edges.measure_plate_box, every
n-th frame) -> windows split where the plate moves -> per window, median line per side over frames where the side
was clearly measured (fuse_boxes; colour veto rejects pipette/hand boundaries). Per plate (same plate, fixed
camera): its size in px is measured from windows where every side was clearly measured; a side hidden for a whole
window (pipette parked on it) is rebuilt from that size, else from the plate aspect ratio measured on all clean
windows (complete_box). Drawn: red = measured sides, orange = rebuilt sides, thin green = YOLO.
Test plates are refused. Output: runs/<ts>_plate-box/{plate_box.mp4, boxes.json, metrics.json, notes.md}

    python scripts/render_plate_box.py --plates 4,1,2,9,10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_all import encode  # noqa: E402
from viz_tracking import OUT_H, OUT_W, title_card  # noqa: E402

from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import clips_for_split, load_splits, plate_of  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.plate_edges import (  # noqa: E402
    box_sides,
    box_size,
    complete_box,
)
from pipette_coach.localizer.plate_grid import measure_clip  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

INSET_W, INSET_H = 560, 400


def draw(frame: np.ndarray, win: dict | None, yolo, title: str, i: int) -> np.ndarray:
    s = OUT_W / frame.shape[1]
    canvas = cv2.resize(frame, (OUT_W, int(frame.shape[0] * s)))
    canvas = cv2.copyMakeBorder(canvas, 0, max(0, OUT_H - canvas.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:OUT_H]

    def poly(img, box, sc, x0, y0, col, th):
        p = np.round(((np.asarray(box) - [x0, y0] + 0.5) * sc - 0.5) * 16).astype(np.int32)
        cv2.polylines(img, [p], True, col, th, cv2.LINE_AA, 4)

    def sides(img, win, sc, x0, y0):
        for k, (a, b) in enumerate(box_sides(win["box"])):
            col = (0, 140, 255) if k in win["rebuilt"] else (0, 0, 255)
            pa = np.round(((a - [x0, y0] + 0.5) * sc - 0.5) * 16).astype(np.int32)
            pb = np.round(((b - [x0, y0] + 0.5) * sc - 0.5) * 16).astype(np.int32)
            cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb), col, 2, cv2.LINE_AA, 4)

    if yolo is not None:
        poly(canvas, yolo, s, 0, 0, (0, 255, 0), 1)
    if win is not None:
        sides(canvas, win, s, 0, 0)
        c = win["box"].mean(0)
        ext = np.ptp(win["box"], axis=0)
        z = 0.8 * min(INSET_W / ext[0], INSET_H / ext[1])
        cw, ch = int(INSET_W / z), int(INSET_H / z)
        x0 = int(np.clip(c[0] - cw / 2, 0, frame.shape[1] - cw))
        y0 = int(np.clip(c[1] - ch / 2, 0, frame.shape[0] - ch))
        crop = cv2.resize(frame[y0 : y0 + ch, x0 : x0 + cw], (INSET_W, INSET_H), interpolation=cv2.INTER_CUBIC)
        sc = INSET_W / cw
        if yolo is not None:
            poly(crop, yolo, sc, x0, y0, (0, 255, 0), 1)
        sides(crop, win, sc, x0, y0)
        cv2.rectangle(crop, (0, 0), (INSET_W - 1, INSET_H - 1), (255, 255, 255), 2)
        canvas[10 : 10 + INSET_H, OUT_W - INSET_W - 10 : OUT_W - 10] = crop
    cv2.rectangle(canvas, (0, OUT_H - 64), (OUT_W, OUT_H), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, OUT_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    info = "red = plate edges (measured)  orange = side hidden all clip, rebuilt from plate size  green = YOLO"
    cv2.putText(canvas, f"frame {i:3d}   {info}", (12, OUT_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="4,1,2,9,10")
    ap.add_argument("--clips", default="", help="only these clips (comma-separated); plate size still from all")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--fps", type=float, default=15)
    a = ap.parse_args()
    cfg = load_config("localizer")
    ec = cfg["plate_edges"]
    det = PlateDetector(cfg)
    splits = load_splits(a.splits)
    only = {c for c in a.clips.split(",") if c}
    run = new_run_dir("plate-box")
    vw = cv2.VideoWriter(str(run / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (OUT_W, OUT_H))
    summary, t0 = {}, time.time()
    plates = [int(x) for x in a.plates.split(",") if x]
    if any(pl in splits["test"] for pl in plates):
        sys.exit("a plate is in the test split; not rendering test data without explicit approval.")
    # pass 1: measure every clip of every plate
    meas, clean = {}, {}
    for pl in plates:
        clips = [c for s in ("train", "val") for c in clips_for_split(s, a.splits) if plate_of(c) == pl]
        meas[pl] = {c: measure_clip(read_frames(c), det, cfg) for c in clips}
        clean[pl] = [box_size(w["box"]) for ws in meas[pl].values() for w in ws
                     if (w["support"] >= ec["min_clean_support"]).all()]  # fmt: skip
        print(f"plate {pl}: measured {len(clips)} clips, {len(clean[pl])} clean windows ({time.time() - t0:.0f}s)")
    # plate aspect ratio measured on every clean window (scale-free, so it pools across plates)
    ratios = [lg / sh for v in clean.values() for sh, lg in v]
    ratio = float(np.median(ratios)) if ratios else None
    if ratios:
        print(f"measured plate aspect ratio {ratio:.4f} (p10 {np.percentile(ratios, 10):.4f}, p90 "
              f"{np.percentile(ratios, 90):.4f}, n={len(ratios)}); SBS footprint 127.76/85.48 = 1.4946")  # fmt: skip
    # pass 2: rebuild hidden sides, render
    for pl in plates:
        size = tuple(float(x) for x in np.median(np.array(clean[pl]), axis=0)) if clean[pl] else None
        spread = np.ptp(np.array(clean[pl]), axis=0).round(1).tolist() if len(clean[pl]) > 1 else None
        print(f"plate {pl}: size {np.round(size, 1) if size else None} px (spread {spread}); "
              f"{'own size' if size else 'aspect ratio'} used for hidden sides")  # fmt: skip
        for clip, wins in meas[pl].items():
            for w in wins:
                w["box"], w["rebuilt"] = complete_box(w["box"], w["support"], size, ratio, cfg)
            summary[clip] = {
                "plate_size_px": size, "aspect_ratio": ratio,
                "windows": [{"start": w["start"], "end": w["end"], "box": w["box"].tolist(),
                             "support": w["support"].round(2).tolist(),
                             "rebuilt_sides": w["rebuilt"]} for w in wins],
            }  # fmt: skip
            if only and clip not in only:
                continue
            rb = sum(len(w["rebuilt"]) for w in wins)
            lines = [clip, f"{len(wins)} static window(s); {rb} side(s) rebuilt (hidden all clip)"]
            for _ in range(int(a.fps // 2)):
                vw.write(title_card(lines))
            for i, f in enumerate(read_frames(clip)):
                win = next((w for w in wins if w["start"] <= i < w["end"]), wins[-1] if wins else None)
                d = det.detect(f)
                vw.write(draw(f, win, None if d is None else d[0], clip, i))
            print(f"  {clip}: windows {len(wins)}, rebuilt sides {rb}", flush=True)
    vw.release()
    encode(run / "raw.mp4", run / "plate_box.mp4")
    (run / "boxes.json").write_text(json.dumps(summary, indent=1))
    n_rb = sum(len(w["rebuilt_sides"]) for v in summary.values() for w in v["windows"])
    n_win = sum(len(v["windows"]) for v in summary.values())
    write_metrics(run, {"clips": len(summary), "windows": n_win, "rebuilt_sides": n_rb})
    (run / "notes.md").write_text(
        "# Plate box from plate edges\n\nBox measured from the plate's own straight edges (no grid homography): "
        "per-frame edge rectangle, median per side over a static window, hidden sides rebuilt from the plate's "
        f"measured size. {len(summary)} clips, {n_win} windows, {n_rb} sides rebuilt. Contains dataset frames: keep "
        "local.\n"
    )
    print(f"done -> {run}")


if __name__ == "__main__":
    main()
