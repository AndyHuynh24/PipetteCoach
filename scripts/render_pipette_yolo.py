"""Render the fine-tuned pipette/tip YOLO on clips (default: all of val plate 4, never trained on) and score the
clip-level well against labels.json.

Per frame: crop around the plate (cached grid, scripts/build_pipette_yolo_dataset.crop_rect) -> YOLO -> boxes;
the best tip box's centre -> canonical plate coords (grid) -> well. Clip prediction = the well with the largest
confidence-weighted count of tip detections. Output: runs/<ts>_pipette-yolo/{pipette_yolo.mp4, metrics.json,
notes.md}. Contains dataset frames: keep local. Test plates refused.

    python scripts/render_pipette_yolo.py [--clips ...] [--weights models/pipette_yolo/weights/best.pt]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pipette_yolo_dataset import crop_rect  # noqa: E402
from cache_plate_grids import load_grid_windows  # noqa: E402
from render_all import encode  # noqa: E402
from viz_tracking import OUT_H, OUT_W, title_card  # noqa: E402

from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import load_splits, plate_of  # noqa: E402
from pipette_coach.localizer.geometry import GRID, project  # noqa: E402
from pipette_coach.localizer.plate_grid import window_at  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.protocol.wells import index_well, nearest_well, well_index, well_str  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tracking.tip_data import clip_wells, split_clips  # noqa: E402

INSET = (620, 460)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="", help="default: every val clip")
    ap.add_argument("--weights", default=str(REPO_ROOT / "models" / "pipette_yolo" / "weights" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fps", type=float, default=15)
    a = ap.parse_args()
    from ultralytics import YOLO

    clips = [c for c in a.clips.split(",") if c] or split_clips("val")
    if any(plate_of(c) in set(load_splits()["test"]) for c in clips):
        sys.exit("test plate clip requested; refusing.")
    model = YOLO(a.weights)
    wells = clip_wells()
    run = new_run_dir("pipette-yolo")
    vw = cv2.VideoWriter(str(run / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (OUT_W, OUT_H))
    rows, ok, nb, n_frames, n_pip, n_tip, t_total = [], 0, 0, 0, 0, 0, 0.0
    for clip in clips:
        frames = read_frames(clip)
        wins = load_grid_windows(clip)
        true = wells[clip]
        votes = np.zeros(96)
        for _ in range(int(a.fps // 2)):
            vw.write(title_card([clip, "dispensed: " + ", ".join(well_str(index_well(k)) for k in true)]))
        for i, f in enumerate(frames):
            w = window_at(wins, i)
            canvas = cv2.resize(f, (OUT_W, int(f.shape[0] * OUT_W / f.shape[1])))
            canvas = cv2.copyMakeBorder(canvas, 0, max(0, OUT_H - canvas.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:OUT_H]
            tip_txt = "no tip"
            if w is not None and w["H_grid"] is not None:
                H = w["H_grid"]
                x0, y0, x1, y1 = crop_rect(H, f.shape)
                crop = f[y0:y1, x0:x1].copy()
                t0 = time.perf_counter()
                r = model.predict(crop, conf=a.conf, imgsz=640, verbose=False)[0]
                t_total += time.perf_counter() - t0
                n_frames += 1
                boxes = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                n_pip += int((cls == 0).any())
                s = INSET[0] / crop.shape[1]
                inset = cv2.resize(crop, (INSET[0], int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
                for gx, gy in project(H, GRID):  # faint grid for reference
                    cv2.circle(inset, (int((gx - x0) * s), int((gy - y0) * s)), 2, (120, 120, 120), -1)
                for k in true:
                    gx, gy = project(H, GRID[k : k + 1])[0]
                    cv2.circle(inset, (int((gx - x0) * s), int((gy - y0) * s)), int(0.5 * s * 14.5), (0, 255, 0), 1)
                for b, c, cf in zip(boxes, cls, conf, strict=True):
                    col = (255, 160, 0) if c == 0 else (0, 0, 255)
                    cv2.rectangle(inset, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), col, 2)
                    cv2.putText(inset, f"{'pipette' if c == 0 else 'tip'} {cf:.2f}", (int(b[0] * s), int(b[1] * s) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)  # fmt: skip
                tips = np.flatnonzero(cls == 1)
                if len(tips):
                    j = tips[np.argmax(conf[tips])]
                    cx, cy = (boxes[j, 0] + boxes[j, 2]) / 2 + x0, (boxes[j, 1] + boxes[j, 3]) / 2 + y0
                    canon = cv2.perspectiveTransform(np.array([[[cx, cy]]], np.float64), np.linalg.inv(H))[0, 0]
                    wl = nearest_well(*canon)
                    n_tip += 1
                    if wl is not None:
                        votes[well_index(wl)] += conf[j]
                        tip_txt = f"tip over {well_str(wl)} ({conf[j]:.2f})"
                    else:
                        tip_txt = f"tip between wells ({conf[j]:.2f})"
                ih = min(inset.shape[0], INSET[1])
                canvas[10 : 10 + ih, OUT_W - INSET[0] - 10 : OUT_W - 10] = inset[:ih]
                cv2.rectangle(canvas, (OUT_W - INSET[0] - 10, 10), (OUT_W - 10, 10 + ih), (255, 255, 255), 2)
            pred = well_str(index_well(int(np.argmax(votes)))) if votes.max() > 0 else "-"
            cv2.rectangle(canvas, (0, OUT_H - 64), (OUT_W, OUT_H), (0, 0, 0), -1)
            cv2.putText(canvas, f"{clip}  frame {i}  {tip_txt}", (12, OUT_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
            true_s = ", ".join(well_str(index_well(k)) for k in true)
            cv2.putText(canvas, f"clip decision so far: {pred}   true: {true_s}"
                        "   (orange = pipette, red = tip, green ring = dispensed well)", (12, OUT_H - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)  # fmt: skip
            vw.write(canvas)
        p = int(np.argmax(votes)) if votes.max() > 0 else None
        hit = p is not None and len(true) == 1 and p == true[0]
        near = p is not None and not hit and len(true) == 1 and max(abs(p // 12 - true[0] // 12),
                                                                    abs(p % 12 - true[0] % 12)) == 1  # fmt: skip
        ok += hit
        nb += near
        rows.append((clip, ", ".join(well_str(index_well(k)) for k in true), "-" if p is None else
                     well_str(index_well(p)), hit))  # fmt: skip
        print(f"{clip}: true {rows[-1][1]} predicted {rows[-1][2]} {'OK' if hit else ''}", flush=True)
    vw.release()
    encode(run / "raw.mp4", run / "pipette_yolo.mp4")
    single = sum(len(wells[c]) == 1 for c in clips)
    m = {"weights": a.weights, "clips": len(clips), "single_well_clips": single, "exact": ok, "off_by_one": nb,
         "frames_with_pipette": n_pip, "frames_with_tip": n_tip, "frames": n_frames,
         "ms_per_frame": 1000 * t_total / max(n_frames, 1)}  # fmt: skip
    write_metrics(run, m)
    lines = ["# Pipette/tip YOLO render", "", f"Exact well {ok}/{single} single-well clips, off-by-one {nb}. Pipette "
             f"box in {n_pip}/{n_frames} frames, tip box in {n_tip}. {m['ms_per_frame']:.0f} ms/frame (plate crop).",
             "", "| clip | true | predicted | ok |", "|---|---|---|---|"]  # fmt: skip
    lines += [f"| {c} | {t} | {p} | {'yes' if h else 'no'} |" for c, t, p, h in rows]
    lines += ["", "Contains dataset frames: keep local."]
    (run / "notes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:3]), "\n->", run / "pipette_yolo.mp4")


if __name__ == "__main__":
    main()
