"""Render a plate-tracking visualisation video (frame-by-frame PlateLocalizer, as the live app would run).

Each clip spec is CLIP or CLIP:cut=F, where cut=F masks the right fraction F of the plate after a
1-second calibration (simulates part of the plate being out of view; always run in live mode).
Output stays in runs/ (dataset frames must not leave the machine).

    python scripts/viz_tracking.py Plate_4_clip_0017 Plate_2_clip_0010 Plate_4_clip_0017:cut=0.4
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
from pipette_coach.localizer import PlateLocalizer, localize_clip_frames  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import GRID, OUTLINE, project  # noqa: E402
from pipette_coach.localizer.session import read_frames, session_prior  # noqa: E402
from pipette_coach.protocol.wells import ROWS  # noqa: E402
from pipette_coach.runs import new_run_dir  # noqa: E402

OUT_W, OUT_H = 1280, 720
INSET_W, INSET_H = 560, 380
COLORS = {"detected": (0, 220, 255), "tracked": (0, 230, 0), "held": (0, 140, 255), "not_found": (0, 0, 255)}  # BGR
CALIB_FRAMES = 30  # 1 s at 30 fps with the full plate visible before the simulated cut


def split_of(clip: str, splits_path: str) -> str:
    try:
        s = load_splits(splits_path)
    except FileNotFoundError:
        return "?"
    return next((k for k, v in s.items() if plate_of(clip) in v), "?")


SUB = 4  # sub-pixel bits for OpenCV drawing


def to_canvas(pts: np.ndarray, scale: float) -> np.ndarray:
    """Image px -> resized-canvas px, matching cv2.resize's pixel-centre convention."""
    return (np.asarray(pts, np.float64) + 0.5) * scale - 0.5


def draw_grid(img: np.ndarray, H: np.ndarray, color: tuple, scale: float = 1.0, labels: bool = False) -> None:
    pts = to_canvas(project(H, GRID), scale)
    outline = np.round(to_canvas(project(H, OUTLINE), scale) * (1 << SUB)).astype(np.int32)
    cv2.polylines(img, [outline], True, (255, 255, 0), 1, cv2.LINE_AA, SUB)
    r = max(2, int(round(1.2 * scale)))
    for k, (x, y) in enumerate(pts):
        c = (int(round(x * (1 << SUB))), int(round(y * (1 << SUB))))
        cv2.circle(img, c, (r if k else r + 3) << SUB, color if k else (255, 0, 255), -1, cv2.LINE_AA, SUB)
    if not labels:
        return
    for i, row in enumerate(ROWS):  # row letters left of column 1
        x, y = to_canvas(project(H, np.array([[-0.9, i]])), scale)[0]
        cv2.putText(img, row, (int(x) - 6, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for c in range(12):  # column numbers above row A
        x, y = to_canvas(project(H, np.array([[c, -0.9]])), scale)[0]
        cv2.putText(
            img, str(c + 1), (int(x) - 6, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )
    x, y = to_canvas(project(H, GRID[:1]), scale)[0]
    cv2.putText(img, "A1", (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)


def compose(
    frame: np.ndarray, fit, title: str, idx: int, ms: float, center: np.ndarray | None, z: float = 2.5
) -> np.ndarray:
    s = OUT_W / frame.shape[1]
    canvas = cv2.resize(frame, (OUT_W, int(frame.shape[0] * s)))
    canvas = cv2.copyMakeBorder(canvas, 0, max(0, OUT_H - canvas.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:OUT_H]
    color = COLORS[fit.status]
    if fit.found:
        draw_grid(canvas, fit.H, color, scale=s)
    # zoom inset around the plate (last known centre so the view stays put when lost)
    if center is not None:
        hw, hh = INSET_W / (2 * z), INSET_H / (2 * z)
        x0, y0 = (
            int(np.clip(center[0] - hw, 0, frame.shape[1] - 2 * hw)),
            int(np.clip(center[1] - hh, 0, frame.shape[0] - 2 * hh)),
        )
        cw = int(2 * hw)
        ch = int(round(cw * INSET_H / INSET_W))  # keep the crop's aspect so one scale applies to x and y
        crop = cv2.resize(frame[y0 : y0 + ch, x0 : x0 + cw], (INSET_W, INSET_H), interpolation=cv2.INTER_CUBIC)
        if fit.found:
            T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
            draw_grid(crop, T @ fit.H, color, scale=INSET_W / cw, labels=True)
        cv2.rectangle(crop, (0, 0), (INSET_W - 1, INSET_H - 1), color, 3)
        canvas[10 : 10 + INSET_H, OUT_W - INSET_W - 10 : OUT_W - 10] = crop
    status = {"detected": "DETECTED", "tracked": "TRACKING", "held": "HELD (memory)", "not_found": "PLATE NOT FOUND"}[
        fit.status
    ]
    info = f"{status}  {fit.method}  wells matched {fit.n_matched}/96  {ms:.0f} ms"
    cv2.rectangle(canvas, (0, OUT_H - 64), (OUT_W, OUT_H), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, OUT_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, f"frame {idx:3d}   {info}", (12, OUT_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )
    return canvas


def title_card(lines: list[str]) -> np.ndarray:
    img = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(
            img, t, (60, 300 + 44 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9 if i else 1.1, (255, 255, 255), 2, cv2.LINE_AA
        )
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--fps", type=float, default=15, help="output fps (dataset is 30 fps -> half speed)")
    ap.add_argument("--no-detector", action="store_true", help="grid-fit fallback instead of the YOLO detector")
    ap.add_argument("--name", default="tracking-viz", help="run folder short name")
    ap.add_argument(
        "--mode",
        choices=["session", "live"],
        default="session",
        help="session: dataset clips of one plate share the remembered orientation and each clip is fitted "
        "on its best frame, then tracked both ways; live: strictly frame by frame (cut= specs always live)",
    )
    a = ap.parse_args()
    cfg = load_config("localizer")
    det = None if a.no_detector else PlateDetector(cfg)
    priors: dict[int, np.ndarray | None] = {}
    run = new_run_dir(a.name)
    raw = run / "tracking_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (OUT_W, OUT_H))
    log = []
    for spec in a.clips:
        clip, _, opt = spec.partition(":")
        cut = float(opt.split("=")[1]) if opt.startswith("cut=") else 0.0
        split = split_of(clip, a.splits)
        if split == "test":
            sys.exit(f"{clip} is a test-split clip; not visualising test data without explicit approval.")
        mode = "live" if cut else a.mode
        desc = f"{clip}  ({split} plate, {mode} mode)" + (
            f"  - right {cut:.0%} of plate masked after 1 s" if cut else ""
        )
        for _ in range(int(a.fps)):
            vw.write(
                title_card(
                    [
                        desc,
                        "magenta = A1, cyan box = plate outline",
                        "yellow = detected, green = tracked, orange = held from memory, red = not found",
                    ]
                )
            )
        loc = PlateLocalizer(cfg, det, rows_toward=plate_rows_toward(plate_of(clip)))
        center, statuses, mask_x, zoom = None, [], None, 2.5
        frames = read_frames(clip)
        if mode == "session":
            plate = plate_of(clip)
            if plate not in priors:  # pass 1 over the plate's clips: remembered orientation
                same = [c for s in ("train", "val") for c in clips_for_split(s, a.splits) if plate_of(c) == plate]
                priors[plate] = session_prior(same, cfg, det)
            t0 = time.perf_counter()
            session_fits = localize_clip_frames(
                frames, cfg, det, prior_H=priors[plate], rows_toward=plate_rows_toward(plate)
            )
            per_frame_ms = (time.perf_counter() - t0) * 1000 / len(frames)
        for i, frame in enumerate(frames):
            if cut and i >= CALIB_FRAMES and loc.H is not None and mask_x is None:
                pr = project(loc.H, OUTLINE)
                mask_x = int(pr[:, 0].max() - cut * (pr[:, 0].max() - pr[:, 0].min()))
            if mask_x is not None:
                frame = frame.copy()
                frame[:, mask_x:] = 0  # simulated: everything right of here is out of view
            if mode == "session":
                fit, ms = session_fits[i], per_frame_ms
            else:
                t0 = time.perf_counter()
                fit = loc.update(frame)
                ms = (time.perf_counter() - t0) * 1000
            if fit.found:
                center = project(fit.H, np.array([[5.5, 3.5]]))[0]
                ext = np.ptp(project(fit.H, OUTLINE), axis=0)  # zoom so the whole plate (+ labels) fits the inset
                zoom = 0.8 * min(INSET_W / ext[0], INSET_H / ext[1])
            statuses.append(fit.status)
            vw.write(compose(frame, fit, desc, i, ms, center, zoom))
        counts = {k: statuses.count(k) for k in ("detected", "tracked", "held", "not_found")}
        log.append(f"{spec}: {counts}")
    vw.release()
    out = run / "tracking.mp4"
    if shutil.which("ffmpeg"):  # re-encode to H.264 so it plays everywhere (QuickTime, browsers)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True)
        raw.unlink()
    else:
        raw.rename(out)
    (run / "notes.md").write_text(
        "# Plate tracking visualisation\n\nFrame-by-frame PlateLocalizer (detect -> track -> re-detect).\n\n"
        + "\n".join(f"- {line}" for line in log)
        + "\n\nContains dataset frames: keep local, do not share or upload.\n"
    )
    print("\n".join(log))
    print("video:", out)


if __name__ == "__main__":
    main()
