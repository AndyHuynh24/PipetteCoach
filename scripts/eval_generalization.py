"""Generalization benchmark: plate + grid under other cameras (scale, rotation, tilt), with known truth.

Val plate 4 clips (never trained on) are re-filmed by random virtual cameras: scale 0.5-2.5x (log-uniform; one
well spacing ~7-36 px), any in-plane rotation, camera tilt up to --max-tilt degrees in a random direction, random
position in a 1920 x 1080 frame. The cached grid (data/processed/plate_grids/) is warped the same way, so the true
grid of every warped frame is known. Per warped clip: plate found? grid error (labelling-free, in pitch) of the
middle frame; correct = error < 0.25 pitch. Pipelines: "fixed" (trained scale only) and "any-scale".

    python scripts/eval_generalization.py [--per-clip 3] [--max-tilt 25]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_plate_grids import load_grid_windows  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import grid_error, project  # noqa: E402
from pipette_coach.localizer.plate_grid import clip_grid_windows, clip_grid_windows_any_scale, window_at  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tracking.tip_data import split_clips  # noqa: E402

OUT_W, OUT_H = 1920, 1080


def camera_warp(centre: np.ndarray, scale: float, rot_deg: float, tilt_deg: float, tilt_dir_deg: float,
                dst_centre: np.ndarray, f: float = 1500.0) -> np.ndarray:  # fmt: skip
    """Image -> new image homography: rotate/scale about the plate centre, tilt the camera (rotation about an
    in-plane axis, pinhole focal f), move the plate centre to dst_centre."""
    T0 = np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]], float)
    a = np.deg2rad(rot_deg)
    RS = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ np.diag([scale, scale, 1])
    axis = np.array([np.cos(np.deg2rad(tilt_dir_deg)), np.sin(np.deg2rad(tilt_dir_deg)), 0.0])
    R, _ = cv2.Rodrigues(axis * np.deg2rad(tilt_deg))
    K = np.diag([f, f, 1.0])
    Tilt = K @ R @ np.linalg.inv(K)
    Tilt = Tilt / Tilt[2, 2]
    T1 = np.array([[1, 0, dst_centre[0]], [0, 1, dst_centre[1]], [0, 0, 1]], float)
    return T1 @ Tilt @ RS @ T0


def run_pipeline(name, frames, det, net, cfg, size):
    if name == "fixed":
        wins = clip_grid_windows(frames, det, net, cfg, size, None)
    else:
        wins, _ = clip_grid_windows_any_scale(frames, det, net, cfg, size, None)
    w = window_at(wins, len(frames) // 2) if wins else None
    return None if w is None else w["H_grid"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", type=int, default=3)
    ap.add_argument("--max-tilt", type=float, default=25.0)
    ap.add_argument("--frames", type=int, default=24, help="frames per warped clip (every 3rd of the clip)")
    ap.add_argument("--pipelines", default="fixed,any-scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detector", default=None, help="plate detector weights (default: configs/localizer.yaml)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cfg = load_config("localizer")
    if a.detector:
        cfg["detector"]["weights"] = a.detector
    det = PlateDetector(cfg)
    net = WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    size = tuple(cfg["plate_edges"]["plate_size_pitch"])
    pipes = [p for p in a.pipelines.split(",") if p]
    rows = []
    t0 = time.time()
    for clip in split_clips("val"):
        wins = load_grid_windows(clip)
        frames_all = read_frames(clip)
        idx = list(range(0, len(frames_all), 3))[: a.frames]
        H0 = window_at(wins, idx[len(idx) // 2])["H_grid"]
        centre = project(H0, np.array([[5.5, 3.5]]))[0]
        for _ in range(a.per_clip):
            s = float(np.exp(rng.uniform(np.log(0.5), np.log(2.5))))
            rot, tilt, tdir = rng.uniform(0, 360), rng.uniform(0, a.max_tilt), rng.uniform(0, 360)
            half = 0.5 * 15.5 * 14.5 * s  # keep the whole plate in the frame
            dst = np.array([rng.uniform(half, OUT_W - half), rng.uniform(half * 0.7, OUT_H - half * 0.7)])
            Wp = camera_warp(centre, s, rot, tilt, tdir, dst)
            frames = [cv2.warpPerspective(frames_all[i], Wp, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE) for i in idx]  # fmt: skip
            H_true = Wp @ H0
            row = {"clip": clip, "scale": s, "tilt": tilt, "rot": rot}
            for p in pipes:
                H = run_pipeline(p, frames, det, net, cfg, size)
                row[p] = None if H is None else grid_error(H_true, H)
            rows.append(row)
            print(f"{clip} s={s:.2f} tilt={tilt:4.1f}: " + "  ".join(
                f"{p} {'-' if row[p] is None else f'{row[p]:.2f}'}" for p in pipes) + f" ({time.time() - t0:.0f}s)",
                flush=True)  # fmt: skip
    metrics = {"n": len(rows), "max_tilt": a.max_tilt}
    head = (
        f"{len(rows)} warped clips: scale 0.5-2.5x, any rotation, tilt 0-{a.max_tilt:.0f} deg. "
        "Correct = grid error < 0.25 pitch (labelling-free)."
    )
    lines = ["# Generalization benchmark (val plate 4, virtual cameras)", "", head, "",
             "| pipeline | bin | found | correct |", "|---|---|---|---|"]  # fmt: skip
    bins = [(0.5, 0.8), (0.8, 1.25), (1.25, 1.8), (1.8, 2.51)]
    for p in pipes:
        err = np.array([np.inf if r[p] is None else r[p] for r in rows])
        sc = np.array([r["scale"] for r in rows])
        metrics[p] = {"found": float(np.isfinite(err).mean()), "correct": float((err < 0.25).mean())}
        for lo, hi in bins:
            m = (sc >= lo) & (sc < hi)
            if m.any():
                found, good = np.isfinite(err[m]).mean(), (err[m] < 0.25).mean()
                lines.append(f"| {p} | scale {lo}-{hi:.2g}x | {found:.0%} | {good:.0%} (n={m.sum()}) |")
                metrics[p][f"correct_{lo}-{hi:.2g}"] = float((err[m] < 0.25).mean())
        tl = np.array([r["tilt"] for r in rows])
        for lo, hi in ((0, 10), (10, 18), (18, 26)):
            m = (tl >= lo) & (tl < hi)
            if m.any():
                lines.append(f"| {p} | tilt {lo}-{hi} deg | {np.isfinite(err[m]).mean():.0%} | "
                             f"{(err[m] < 0.25).mean():.0%} (n={m.sum()}) |")  # fmt: skip
    run = new_run_dir("generalization")
    write_metrics(run, metrics)
    (run / "notes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
