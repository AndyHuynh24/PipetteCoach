"""Render the plate box reconstructed from the wells + measured plate edges (localizer/wells_box.py). No colour,
no standard plate dimensions, no grid labels.

Per clip: static windows + edge box per window (render_plate_box.measure_clip) -> well rings detected on the same
sampled frames (--wells rings: hand-made clean-ring detector; --wells net: heatmap net) -> pooled over the window ->
lattice homography. Per plate: size in pitch units (long, short) from its clean windows (every side measured);
plates without one use the median of the others (scale-free). Side colours: red = measured edge, orange = hidden
edge placed from the plate size, magenta = from the outer wells. Cyan dots = pooled well detections.
Test plates are refused. Output: runs/<ts>_wells-box/{wells_box.mp4, boxes.json, metrics.json, notes.md}

    python scripts/render_wells_box.py --plates 1,2 --wells rings
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
from render_plate_box import INSET_H, INSET_W, measure_clip  # noqa: E402
from viz_tracking import OUT_H, OUT_W, title_card  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import clips_for_split, load_splits, plate_of, plate_rows_toward  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import orient  # noqa: E402
from pipette_coach.localizer.openings import detect_rings  # noqa: E402
from pipette_coach.localizer.plate_edges import box_sides  # noqa: E402
from pipette_coach.localizer.rings import in_polygon  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.localizer.wells_box import (  # noqa: E402
    box_from_wells,
    fit_lattice,
    grid_from_lattice,
    plate_size_pitch,
    pool_points,
    snap_frame,
)
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

SRC_COL = {"edge": (0, 0, 255), "size": (0, 140, 255), "wells": (255, 0, 255), "edge-only": (0, 0, 255)}
SIDE_NAMES = ["uL", "vT", "uR", "vB"]  # box_from_wells corners (uL,vT),(uR,vT),(uR,vB),(uL,vB); side k = k-1 -> k


def grow(quad: np.ndarray, pitch: float, k: float) -> np.ndarray:
    c = quad.mean(0)
    return c + (quad - c) * (1 + k * pitch / np.linalg.norm(quad - c, axis=1, keepdims=True))


def detect_wells(frame: np.ndarray, quad: np.ndarray, cfg: dict, net) -> np.ndarray:
    p = cfg["wells"]["pitch_px_nominal"]
    if net is None:
        return detect_rings(frame, quad, cfg)
    g = grow(np.asarray(quad, np.float64), p, 0.5).astype(np.float32)
    x0, y0 = np.floor(g.min(0)).astype(int)
    x1, y1 = np.ceil(g.max(0)).astype(int)
    P, _ = net.detect(frame, (int(x0), int(y0), int(x1), int(y1)))
    return P[in_polygon(P, g)] if len(P) else P


def draw(frame: np.ndarray, win: dict | None, title: str, i: int, wells=None) -> np.ndarray:
    """wells = (matched (96,), positions (96, 2)) from snap_frame: green = detected this frame, yellow ring =
    reconstructed (hidden), magenta = A1."""
    s = OUT_W / frame.shape[1]
    canvas = cv2.resize(frame, (OUT_W, int(frame.shape[0] * s)))
    canvas = cv2.copyMakeBorder(canvas, 0, max(0, OUT_H - canvas.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:OUT_H]

    def P(x, sc, x0, y0):
        return tuple(int(v) for v in np.round(((np.asarray(x) - [x0, y0] + 0.5) * sc - 0.5) * 16))

    def overlay(img, sc, x0, y0, dots):
        if wells is not None:
            matched, pos = wells
            r = 3 if dots else 1
            for k in range(len(pos)):
                c = P(pos[k], sc, x0, y0)
                if k == 0:
                    cv2.circle(img, c, (r + 2) << 4, (255, 0, 255), -1, cv2.LINE_AA, 4)
                elif matched[k]:
                    cv2.circle(img, c, r << 4, (0, 255, 0), -1, cv2.LINE_AA, 4)
                else:
                    cv2.circle(img, c, (r + 1) << 4, (0, 255, 255), 1, cv2.LINE_AA, 4)
        for k, (a, b) in enumerate(box_sides(win["box"])):
            cv2.line(img, P(a, sc, x0, y0), P(b, sc, x0, y0), SRC_COL[win["src"][k]], 2, cv2.LINE_AA, 4)

    if win is not None:
        overlay(canvas, s, 0, 0, False)
        c, ext = win["box"].mean(0), np.ptp(win["box"], axis=0)
        z = 0.8 * min(INSET_W / ext[0], INSET_H / ext[1])
        cw, ch = int(INSET_W / z), int(INSET_H / z)
        x0 = int(np.clip(c[0] - cw / 2, 0, frame.shape[1] - cw))
        y0 = int(np.clip(c[1] - ch / 2, 0, frame.shape[0] - ch))
        crop = cv2.resize(frame[y0 : y0 + ch, x0 : x0 + cw], (INSET_W, INSET_H), interpolation=cv2.INTER_CUBIC)
        overlay(crop, INSET_W / cw, x0, y0, True)
        cv2.rectangle(crop, (0, 0), (INSET_W - 1, INSET_H - 1), (255, 255, 255), 2)
        canvas[10 : 10 + INSET_H, OUT_W - INSET_W - 10 : OUT_W - 10] = crop
    cv2.rectangle(canvas, (0, OUT_H - 64), (OUT_W, OUT_H), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, OUT_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    info = "box: red = edge, orange = hidden side from size   wells: green = detected, yellow = reconstructed, A1"
    cv2.putText(canvas, f"frame {i:3d}   {info}", (12, OUT_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="1,2")
    ap.add_argument("--wells", choices=["rings", "net"], default="rings")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--render", default="", help="plates to render (default: all of --plates); others only measured")
    ap.add_argument("--max-measure-only", type=int, default=6, help="clips measured per measure-only plate")
    a = ap.parse_args()
    cfg = load_config("localizer")
    ec = cfg["plate_edges"]
    pitch = cfg["wells"]["pitch_px_nominal"]
    det = PlateDetector(cfg)
    net = None
    if a.wells == "net":
        from pipette_coach.localizer.well_net import WellNet

        net = WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    splits = load_splits(a.splits)
    plates = [int(x) for x in a.plates.split(",") if x]
    if any(pl in splits["test"] for pl in plates):
        sys.exit("a plate is in the test split; not rendering test data without explicit approval.")
    render = {int(x) for x in a.render.split(",") if x} or set(plates)
    run = new_run_dir(f"wells-box-{a.wells}")
    t0 = time.time()
    # pass 1: edges + wells per static window
    data = {}
    for pl in plates:
        clips = [c for s in ("train", "val") for c in clips_for_split(s, a.splits) if plate_of(c) == pl]
        if pl not in render:
            clips = clips[: a.max_measure_only]  # measured only for the plate size
        for clip in clips:
            frames = read_frames(clip)
            wins = measure_clip(frames, det, cfg)
            for w in wins:
                pts = []
                for i in range(w["start"], min(w["end"], len(frames)), ec["every_n"]):
                    d = det.detect(frames[i])
                    if d is not None:
                        pts.append(detect_wells(frames[i], d[0], cfg, net))
                w["wells"] = pool_points(pts, pitch, min_frames=2 if len(pts) >= 4 else 1)
                lat = fit_lattice(w["wells"], tuple(cfg["wells"]["pitch_px_range"]))
                w["lat"] = lat
                clean = (w["support"] >= ec["min_clean_support"]).all() and lat is not None
                w["size"] = plate_size_pitch(lat[0], w["box"], lat[1]) if clean else None
            data[clip] = (pl, wins)
        print(f"plate {pl}: {len(clips)} clips measured ({time.time() - t0:.0f}s)", flush=True)
    sizes = {pl: [w["size"] for c, (p, ws) in data.items() if p == pl for w in ws if w["size"]] for pl in plates}

    def med(v: list, min_n: int = 1) -> tuple[float, float] | None:  # per-axis median over clean axes
        arr = np.array(v, dtype=float).reshape(-1, 2)
        if len(arr) == 0 or ((~np.isnan(arr)).sum(axis=0) < min_n).any():
            return None
        return tuple(float(x) for x in np.nanmedian(arr, axis=0))

    size_all = med([s for v in sizes.values() for s in v])
    print(f"all-plate size (long, short) pitch {np.round(size_all, 3) if size_all else None}")
    metrics = {"wells": a.wells, "plate_size_pitch": {}, "all_plate_size_pitch": size_all}
    for pl in plates:
        arr = np.array(sizes[pl], dtype=float).reshape(-1, 2)
        n_ok = (~np.isnan(arr)).sum(0).tolist()
        sd = np.nanstd(arr, axis=0).round(3).tolist() if len(arr) else None
        m = med(sizes[pl])
        print(f"plate {pl}: size (long, short) pitch {np.round(m, 3) if m else None} from {n_ok} clean axes, sd {sd}")
        metrics["plate_size_pitch"][pl] = m
    # pass 2: boxes + render
    vw = cv2.VideoWriter(str(run / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (OUT_W, OUT_H))
    summary, n_src = {}, {"edge": 0, "size": 0, "wells": 0, "edge-only": 0}
    gate = cfg["plate_edges"]["well_gate_pitch"]
    n_det: list[int] = []
    for clip, (pl, wins) in data.items():
        if pl not in render:
            continue
        size = med(sizes[pl], 3) or size_all  # own size (>= 3 clean, symmetric measurements per axis), else all
        for w in wins:
            r = None
            if w["lat"] is not None:
                r = box_from_wells(w["lat"][0], w["lat"][1], w["box"], w["support"], size, (12, 8), cfg)
            w["H_grid"] = None
            if r is None:  # no usable lattice: the edge box as measured
                w["src"] = ["edge-only"] * 4
            else:
                w["box"] = r[0]
                w["src"] = [r[1][n] for n in SIDE_NAMES]
                # 12 x 8 grid from the lattice, centred in the box; refined once on the pooled wells (stable within
                # the window); A1 by the per-plate orientation rule
                Hg = grid_from_lattice(w["lat"][0], w["lat"][1], r[2]["rect"], r[2]["u_long"], (12, 8))
                Hg = snap_frame(Hg, w["wells"], gate)[0]
                oc = cfg["orientation"]
                w["H_grid"] = orient(Hg, oc["a1_rule"], oc["camera_mirrored"], rows_toward=plate_rows_toward(pl))
            for x in w["src"]:
                n_src[x] += 1
        summary[clip] = [{"start": w["start"], "end": w["end"], "box": np.asarray(w["box"]).tolist(), "sides": w["src"],
                          "n_wells": len(w["wells"])} for w in wins]  # fmt: skip
        srcs = ", ".join(f"[{'/'.join(w['src'])}]" for w in wins)
        for _ in range(int(a.fps // 2)):
            vw.write(title_card([clip, f"{len(wins)} window(s); sides (L,T,R,B in lattice): {srcs}"]))
        for i, f in enumerate(read_frames(clip)):
            win = next((w for w in wins if w["start"] <= i < w["end"]), wins[-1] if wins else None)
            wells = None
            if win is not None and win.get("H_grid") is not None:
                dets = detect_wells(f, win["box"], cfg, net)
                _, matched, pos = snap_frame(win["H_grid"], dets, gate, min_matches=10**9)  # match only, no refit
                wells = (matched, pos)
                n_det.append(int(matched.sum()))
            vw.write(draw(f, win, clip, i, wells))
        print(f"  {clip}: {srcs}", flush=True)
    vw.release()
    encode(run / "raw.mp4", run / "wells_box.mp4")
    (run / "boxes.json").write_text(json.dumps(summary, indent=1))
    write_metrics(run, {**metrics, "side_sources": n_src, "clips": len(summary),
                        "detected_wells_per_frame_median": float(np.median(n_det)) if n_det else None})  # fmt: skip
    (run / "notes.md").write_text(
        f"# Plate box from wells + edges ({a.wells})\n\nSide sources: {n_src}. Plate size (pitch units) per plate: "
        f"{metrics['plate_size_pitch']}. No colour, no standard dimensions, no labels. Contains dataset frames: keep "
        "local.\n"
    )
    print(f"done -> {run}  side sources {n_src}")


if __name__ == "__main__":
    main()
