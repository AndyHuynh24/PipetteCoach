"""Render the plate box + 12 x 8 well grid (localizer/plate_grid.py, the approved method) on any video.

The video is processed in chunks (default 3 s). Per chunk the plate detector gives the plate's size, frames are
rescaled so a well spacing is ~wells.pitch_px_nominal px (the size the detectors were trained at; the plate is
searched at several scales for closer cameras), the grid is
fitted, and results are mapped back to the original frames. Per frame: well-net detections snapped to the grid
(green = detected, yellow = reconstructed, magenta = A1), box sides (red = measured edge, orange = rebuilt).
Output: runs/<ts>_video-grid/video_grid.mp4 (keep local: third-party video).

    python scripts/render_video.py --video data/raw/pipette1.mp4 --seconds 60
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_all import encode  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import GRID, project  # noqa: E402
from pipette_coach.localizer.plate_edges import box_sides  # noqa: E402
from pipette_coach.localizer.plate_grid import clip_grid_windows_any_scale, detect_wells_net, window_at  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.localizer.wells_box import snap_frame  # noqa: E402
from pipette_coach.localizer.wells_first import find_plate, plate_grid  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

SIDE_COL = {"edge": (0, 0, 255), "size": (0, 140, 255), "wells": (255, 0, 255), "edge-only": (0, 0, 255)}


def draw(frame: np.ndarray, win: dict | None, wells, title: str) -> np.ndarray:
    img = frame.copy()
    lw = max(1, int(round(frame.shape[1] / 900)))
    if win is not None:
        for k, (a, b) in enumerate(box_sides(win["box"])):
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                     SIDE_COL[win["sides"][k]], 2 * lw, cv2.LINE_AA)  # fmt: skip
    if wells is not None:
        matched, pos = wells
        r = max(2, int(round(np.linalg.norm(pos[1] - pos[0]) * 0.18)))
        for k, (x, y) in enumerate(pos):
            c = (int(round(x)), int(round(y)))
            if k == 0:
                cv2.circle(img, c, r + 2, (255, 0, 255), -1, cv2.LINE_AA)
            elif matched[k]:
                cv2.circle(img, c, r, (0, 255, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, r + 1, (0, 255, 255), lw, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34 * lw), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 24 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * lw, (255, 255, 255), lw, cv2.LINE_AA)
    return img


def read_chunks(a, fps: float):
    """Yield the video's frames chunk by chunk (wide frames downscaled to --max-width)."""
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
    n_total, n_chunk, done = int(a.seconds * fps), max(10, int(a.chunk * fps)), 0
    while done < n_total:
        frames = []
        while len(frames) < min(n_chunk, n_total - done):
            ok, f = cap.read()
            if not ok:
                break
            if f.shape[1] > a.max_width:
                h = int(f.shape[0] * a.max_width / f.shape[1])
                f = cv2.resize(f, (a.max_width, h), interpolation=cv2.INTER_AREA)
            frames.append(f)
        if not frames:
            return
        yield done, frames
        done += len(frames)


def on_grid_share(pts: np.ndarray, H: np.ndarray) -> float:
    """Share of detected well centres (original px) that sit on a node of grid H (within 0.3 pitch)."""
    if len(pts) == 0:
        return 0.0
    from pipette_coach.localizer.geometry import pitch_px

    d = np.linalg.norm(pts[:, None] - project(H, GRID)[None], axis=2).min(1)
    return float((d < 0.3 * pitch_px(H)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--chunk", type=float, default=3.0)
    ap.add_argument("--max-width", type=int, default=1920, help="downscale wider frames first")
    ap.add_argument("--detector", default=None, help="plate detector weights (default: configs/localizer.yaml)")
    a = ap.parse_args()
    cfg = load_config("localizer")
    if a.detector:
        cfg["detector"]["weights"] = a.detector
    det = PlateDetector(cfg)
    net = WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    size = tuple(cfg["plate_edges"]["plate_size_pitch"])
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30
    t0 = time.time()
    # pass 1: wells-first plate per chunk (scale, orientation, and the full plate when all 96 wells are seen)
    found = [find_plate(frames, det, net, cfg) for _, frames in read_chunks(a, fps)]
    full = [k for k, r in enumerate(found) if r is not None and r["full"]]
    grids = {k: plate_grid(found[k], cfg, None) for k in full}
    print(f"pass 1: {len(full)}/{len(found)} chunks with the full plate ({time.time() - t0:.0f}s)", flush=True)
    run = new_run_dir("video-grid")
    vw = None
    stats = {"frames": 0, "with_grid": 0, "chunks": len(found), "full_plate": len(full), "memory": 0, "fallback": 0,
             "no_plate": 0}  # fmt: skip
    n_det = []
    for k, (done, frames) in enumerate(read_chunks(a, fps)):
        r = found[k]
        wins, s, src = [], None, ""
        if k in grids:
            src = "wells-first (full plate)"
            H, s, box = grids[k], r["scale"], r["outline"]
        else:
            # memory: the nearest full-plate chunk, if this chunk's wells sit on its grid (the plate did not move)
            H = None
            for j in sorted(full, key=lambda j: abs(j - k)):
                if r is not None and len(r["wells"]) >= 10:
                    pts = r["wells"] / r["scale_fit"]
                else:
                    sj = found[j]["scale"]
                    fs = cv2.resize(frames[len(frames) // 2], None, fx=sj, fy=sj, interpolation=cv2.INTER_AREA)
                    pts = detect_wells_net(fs, project(grids[j], GRID[[0, 11, 95, 84]]) * sj, cfg, net) / sj
                if len(pts) >= 10 and on_grid_share(pts, grids[j]) >= 0.7:
                    H, s, box, src = grids[j], found[j]["scale"], found[j]["outline"], f"memory (chunk {j})"
                    stats["memory"] += 1
                    break
                if abs(j - k) > 3:
                    break
        if H is not None:
            wins = [{"start": 0, "end": len(frames), "box": box, "sides": ["wells"] * 4, "H_grid": H,
                     "Hs": np.diag([s, s, 1.0]) @ H}]  # fmt: skip
        else:
            wins, s = clip_grid_windows_any_scale(frames, det, net, cfg, size, None,
                                                  scale=None if r is None else r["scale"])  # fmt: skip
            wins = [{**w, "Hs": w["H_scaled"]} for w in wins]
            src = "fallback (edges + detector box)"
            stats["fallback" if wins else "no_plate"] += 1
        for i, f in enumerate(frames):
            w = window_at(wins, i) if wins else None
            wells = None
            if w is not None and w["H_grid"] is not None:
                fs_i = f if abs(s - 1) < 0.05 else cv2.resize(f, None, fx=s, fy=s)
                dets = detect_wells_net(fs_i, project(w["Hs"], GRID[[0, 11, 95, 84]]), cfg, net) / s
                _, matched, pos = snap_frame(w["H_grid"], dets, cfg["plate_edges"]["well_gate_pitch"], 10**9)
                wells = (matched, pos)
                stats["with_grid"] += 1
                n_det.append(int(matched.sum()))
            t = (done + i) / fps + a.start
            if wells is not None:
                info = (f"t={t:5.1f}s  {int(wells[0].sum())}/96 wells detected (green), rest reconstructed "
                        f"(yellow), A1 magenta   [{src}]")  # fmt: skip
            else:
                info = f"t={t:5.1f}s  no plate grid"
            img = draw(f, w if wells is not None else None, wells, info)
            if vw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(str(run / "raw.mp4"), fourcc, fps, (img.shape[1], img.shape[0]))
            vw.write(img)
        stats["frames"] = done + len(frames)
        print(f"chunk {k}: {src or 'no plate'} ({time.time() - t0:.0f}s)", flush=True)
    if vw is not None:
        vw.release()
        encode(run / "raw.mp4", run / "video_grid.mp4")
    stats["median_detected_wells"] = float(np.median(n_det)) if n_det else None
    write_metrics(run, {"video": a.video, "start_s": a.start, "seconds": a.seconds, **stats})
    (run / "notes.md").write_text(f"# Plate grid on {a.video}\n\nFirst {a.seconds:.0f} s from {a.start:.0f} s. "
                                  f"{stats}. Third-party video: keep local.\n")  # fmt: skip
    print("->", run / "video_grid.mp4", stats)


if __name__ == "__main__":
    main()
