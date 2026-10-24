"""Well-detector visualisation: plate box (YOLO, every frame) + radial-edge circles that are wells.

Circles come from the radial-edge detector inside the box's well area. Only circles that land on one of the
96 well positions of the localizer's fitted plate grid are drawn (nearest circle per well; no dataset
labels are used); every other circle is counted in the status line but not drawn. A homography is then fit
on the detected well circles alone and every missing well is reconstructed from it (blue), with the fit's
RMS and leave-one-out reconstruction error shown on screen. Colour = radial-edge
score (red = barely above threshold, green = perfect circle). Right panel: the score heatmap.
Spec CLIP:fill=KIND paints SYNTHETIC liquid into the wells first (dye, milky,
dark, half, mixed) using the localizer's grid only to know where to paint - clearly marked in the video.
Output stays in runs/ (dataset frames must not leave the machine).

    python scripts/viz_wells.py Plate_4_clip_0017 Plate_4_clip_0017:fill=mixed Plate_2_clip_0010
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import clips_for_split, load_splits, plate_of, plate_rows_toward  # noqa: E402
from pipette_coach.localizer import localize_clip_frames  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import GRID, OUTLINE, pitch_px, project  # noqa: E402
from pipette_coach.localizer.rings import plate_circles, radial_edge_response  # noqa: E402
from pipette_coach.localizer.session import read_frames, session_prior  # noqa: E402
from pipette_coach.localizer.synthetic import paint_fill  # noqa: E402
from pipette_coach.runs import new_run_dir  # noqa: E402

PANEL_W, PANEL_H, BAR = 640, 440, 60


def to_panel(pts: np.ndarray, x0: int, y0: int, s: float) -> np.ndarray:
    return (pts - [x0, y0] + 0.5) * s - 0.5


def reconstruct(
    canon: np.ndarray, pts: np.ndarray, scores: np.ndarray, H_grid: np.ndarray, rng_seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict | None]:
    """Fit a homography (canonical well -> image) on the detected well circles (RANSAC), keep the inlier
    circles as measured wells and reconstruct every other well from the fit.

    Stats: RMS of the inlier circles, and leave-one-out error (predict a detected well from a fit on the
    others; median over up to 24 wells) = the expected error of a reconstructed well."""
    p = pitch_px(H_grid)
    if len(pts) < 8:
        return pts, scores, np.zeros((0, 2)), None
    Hc, inl = cv2.findHomography(canon.astype(np.float32), pts.astype(np.float32), cv2.RANSAC, 0.12 * p)
    if Hc is None or inl.sum() < 8:
        return pts, scores, np.zeros((0, 2)), None
    inl = inl.ravel().astype(bool)
    res = np.linalg.norm(project(Hc, canon[inl]) - pts[inl], axis=1)
    rng = np.random.default_rng(rng_seed)
    idx = np.nonzero(inl)[0]
    loo = []
    for k in rng.choice(idx, size=min(24, len(idx)), replace=False):
        others = idx[idx != k]
        Hk, _ = cv2.findHomography(canon[others].astype(np.float32), pts[others].astype(np.float32), 0)
        if Hk is not None:
            loo.append(float(np.linalg.norm(project(Hk, canon[k : k + 1])[0] - pts[k])))
    have = {tuple(c) for c in canon[inl].astype(int)}
    missing = np.array([g for g in GRID if tuple(g.astype(int)) not in have], np.float32).reshape(-1, 2)
    recon = project(Hc, missing) if len(missing) else np.zeros((0, 2))
    stats = {"n_fit": int(inl.sum()), "rms": float(np.sqrt(np.mean(res**2))), "loo": float(np.median(loo))}
    return pts[inl], scores[inl], recon, stats


def render(
    frame: np.ndarray, roi: tuple, cfg: dict, title: str, synthetic: bool, quad: np.ndarray | None, H: np.ndarray | None
) -> np.ndarray:
    x0, y0, x1, y1 = roi
    crop = frame[y0:y1, x0:x1]
    t0 = time.perf_counter()
    ratio, _ = radial_edge_response(crop, cfg)
    P, S, n_outside = (np.zeros((0, 2)), np.zeros(0), 0) if quad is None else plate_circles(frame, quad, cfg)
    wells, n_other, recon, stats = np.zeros((0, 2)), len(P), np.zeros((0, 2)), None
    if H is not None and len(P):
        # a circle is a well if it lands on one of the 96 grid positions (nearest circle per well)
        grid = project(H, GRID)
        d = np.linalg.norm(grid[:, None] - P[None], axis=2)
        j = d.argmin(1)
        on = d[np.arange(len(grid)), j] < 0.3 * pitch_px(H)
        n_other = len(P) - len(np.unique(j[on]))
        wells, S, recon, stats = reconstruct(GRID[on], P[j[on]], S[j[on]], H)
    ms = (time.perf_counter() - t0) * 1000
    s = min(PANEL_W / crop.shape[1], PANEL_H / crop.shape[0])
    size = (int(crop.shape[1] * s), int(crop.shape[0] * s))
    left = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.clip((ratio - 0.3) / 0.6 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, size, interpolation=cv2.INTER_NEAREST)
    lo = cfg["wells"]["radial_edge"]["min_ratio"]
    r_open = 0.44 * cfg["wells"]["pitch_px_nominal"] * s
    for (px, py), sc in zip(to_panel(wells, x0, y0, s), S / 100.0, strict=True):
        f = float(np.clip((sc - lo) / (0.85 - lo), 0, 1))
        col = (0, int(255 * f), int(255 * (1 - f)))
        c = (int(round(px * 16)), int(round(py * 16)))
        cv2.circle(left, c, int(r_open * 16), col, 1, cv2.LINE_AA, 4)
        cv2.circle(left, c, 2 * 16, col, -1, cv2.LINE_AA, 4)
        cv2.circle(heat, c, 2 * 16, (255, 255, 255), -1, cv2.LINE_AA, 4)
    for px, py in to_panel(recon, x0, y0, s):  # missing wells, reconstructed from the homography
        c = (int(round(px * 16)), int(round(py * 16)))
        cv2.circle(left, c, int(r_open * 16), (255, 120, 0), 1, cv2.LINE_AA, 4)
        cv2.drawMarker(left, (int(round(px)), int(round(py))), (255, 120, 0), cv2.MARKER_CROSS, 6, 1)
    if quad is not None:
        q16 = np.round(to_panel(np.asarray(quad, np.float64), x0, y0, s) * 16).astype(np.int32)
        cv2.polylines(left, [q16], True, (255, 255, 0), 1, cv2.LINE_AA, 4)
    if stats is not None:
        txt = (
            f"H from {stats['n_fit']} circles: RMS {stats['rms']:.2f}px, leave-one-out {stats['loo']:.2f}px"
            f" | {len(recon)} reconstructed (blue)"
        )
        cv2.rectangle(left, (0, size[1] - 22), (size[0], size[1]), (0, 0, 0), -1)
        cv2.putText(left, txt, (6, size[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 120), 1, cv2.LINE_AA)
    canvas = np.zeros((PANEL_H + BAR, 2 * PANEL_W, 3), np.uint8)
    canvas[: size[1], : size[0]] = left
    canvas[: size[1], PANEL_W : PANEL_W + size[0]] = heat
    cv2.putText(canvas, title, (10, PANEL_H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    box = "no plate box" if quad is None else f"{n_outside} outside box"
    info = f"{len(wells)}/96 wells measured, {len(recon)} reconstructed | hidden: {n_other} non-well circles, {box}"
    info += f" | {ms:.0f} ms"
    cv2.putText(canvas, info, (10, PANEL_H + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2, cv2.LINE_AA)
    if synthetic:
        cv2.putText(canvas, "SYNTHETIC FILL", (PANEL_W - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(canvas, "radial-edge score", (PANEL_W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--plates", default="", help="comma-separated plates: render all their clips (train/val only)")
    ap.add_argument("--name", default="wells-viz")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config("localizer")  # the pipeline as configured; plate_circles uses the radial-edge finder itself
    det = PlateDetector(cfg)
    splits = load_splits(a.splits)
    specs = list(a.clips)
    for pl in [int(x) for x in a.plates.split(",") if x]:
        specs += [c for s_ in ("train", "val") for c in clips_for_split(s_, a.splits) if plate_of(c) == pl]
    priors: dict[int, np.ndarray | None] = {}
    run = new_run_dir(a.name)
    raw = run / "wells_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (2 * PANEL_W, PANEL_H + BAR))
    log = []
    for spec in specs:
        clip, _, opt = spec.partition(":")
        fill = opt.split("=")[1] if opt.startswith("fill=") else None
        if plate_of(clip) in splits["test"]:
            sys.exit(f"{clip} is a test-split clip; not visualising test data without explicit approval.")
        frames = read_frames(clip)
        plate = plate_of(clip)
        if plate not in priors:  # session memory: the plate's orientation from its other clips
            same = [c for s_ in ("train", "val") for c in clips_for_split(s_, a.splits) if plate_of(c) == plate]
            priors[plate] = session_prior(same, cfg, det)
        fits = localize_clip_frames(frames, cfg, det, prior_H=priors[plate], rows_toward=plate_rows_toward(plate))
        Hs = [f.H for f in fits if f.found]
        title_card = np.zeros((PANEL_H + BAR, 2 * PANEL_W, 3), np.uint8)
        cv2.putText(
            title_card, spec, (40, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
        )
        if not Hs:
            cv2.putText(
                title_card, "plate not found in any frame", (40, PANEL_H // 2 + 50), 0, 0.9, (0, 0, 255), 2, cv2.LINE_AA
            )
            for _ in range(int(a.fps)):
                vw.write(title_card)
            log.append(f"{spec}: plate not found in any frame")
            continue
        for _ in range(int(a.fps // 2)):
            vw.write(title_card)
        H = Hs[len(Hs) // 2]
        p = pitch_px(H)
        q = project(H, OUTLINE)
        roi = tuple(int(v) for v in np.r_[q.min(0) - 1.0 * p, q.max(0) + 1.0 * p])
        roi = (max(roi[0], 0), max(roi[1], 0), roi[2], roi[3])
        title = f"{clip}" + (f"  fill={fill} (SYNTHETIC)" if fill else "  (real frames)")
        quad = None
        for i, f in enumerate(frames):
            # same seed every frame -> the same wells get the same synthetic liquid throughout the clip
            img = paint_fill(f, fits[i].H if fits[i].found else H, fill, np.random.default_rng(a.seed)) if fill else f
            d = det.detect(img)  # plate box on every frame (keep the last one if this frame has none)
            quad = d[0] if d is not None else quad
            Hf = fits[i].H if fits[i].found else None
            vw.write(render(img, roi, cfg, f"{title}   frame {i}", fill is not None, quad, Hf))
        log.append(f"{spec}: {len(frames)} frames, plate in {sum(f.found for f in fits)}")
    vw.release()
    out = run / "wells.mp4"
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True)
        raw.unlink()
    else:
        raw.rename(out)
    (run / "notes.md").write_text(
        "# Well-detector visualisation (plate box + radial-edge circles that are wells)\n\n"
        "Drawn: radial-edge circles inside the YOLO plate box that land on the localizer's grid positions (no "
        "dataset labels). Other circles are counted, not drawn. Fills marked SYNTHETIC are rendered, not real.\n\n"
        + "\n".join(f"- {x}" for x in log)
        + "\n\nContains dataset frames: keep local.\n"
    )
    print("\n".join(log))
    print("video:", out)


if __name__ == "__main__":
    main()
