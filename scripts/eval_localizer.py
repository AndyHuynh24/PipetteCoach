"""Evaluate the plate localizer on val clips. Writes runs/<ts>_localizer-val/{metrics.json, notes.md}.

1. Per clip (detector + snap, best frame): found?, grid error vs the auto-label reference (confident
   global grid fit on the same clip; checked by eye during exploration - NOT human labels).
2. A1 consistency: one plate per split here, so A1 should land in the same image spot in every clip.
3. Cut-plate test: crop the plate so part of it is outside the image (6 scenarios), then
   (a) detector must refuse (border guard) rather than return a shifted grid,
   (b) tracking from a perturbed calibration must recover the grid from the visible wells.
4. Latency per stage.

    python scripts/eval_localizer.py --split val
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clip_path, clips_for_split  # noqa: E402
from pipette_coach.localizer import PlateLocalizer, localize_clip  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import GRID, grid_error, pitch_px, project, quad_to_H  # noqa: E402
from pipette_coach.localizer.refine import snap_to_wells, track  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

CUTS = {  # fraction of the plate's image bbox kept inside the frame: (x0, y0, x1, y1)
    "right_40pct_cut": (0, 0, 0.6, 1),
    "left_40pct_cut": (0.4, 0, 1, 1),
    "bottom_50pct_cut": (0, 0, 1, 0.5),
    "top_50pct_cut": (0, 0.5, 1, 1),
    "corner_60x50pct": (0, 0, 0.6, 0.5),
    "corner_35x40pct": (0.65, 0.6, 1, 1),
}
CORRECT = 0.25  # a grid is "correct" if mean well-centre error < 1/4 pitch (no well can be confused)


def read_frames(clip: str, every: int) -> list[tuple[int, np.ndarray]]:
    cap, out, i = cv2.VideoCapture(str(clip_path(clip))), [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        if i % every == 0:
            out.append((i, f))
        i += 1


def perturb(H: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Plate bumped since the last frame: up to 0.3 pitch shift and 2 degrees rotation."""
    p, c = pitch_px(H), project(H, np.array([[5.5, 3.5]]))[0]
    a, t = np.deg2rad(rng.uniform(-2, 2)), rng.uniform(-0.3, 0.3, 2) * p
    T1 = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    T2 = np.array([[1, 0, c[0] + t[0]], [0, 1, c[1] + t[1]], [0, 0, 1]])
    return T2 @ R @ T1 @ H


def cut_canvas(img: np.ndarray, H: np.ndarray, keep: tuple[float, float, float, float]) -> np.ndarray:
    """Keep only part of the plate (plus 3 pitches of bench on uncut sides) at the original scale;
    everything else becomes bench colour, so the plate edge on the cut side is the image edge."""
    pr = project(H, GRID)
    (x0, y0), (x1, y1), m = pr.min(0), pr.max(0), 3 * pitch_px(H)
    a, b, c, d = keep
    X0 = int(x0 + a * (x1 - x0)) if a > 0 else int(max(x0 - m, 0))
    Y0 = int(y0 + b * (y1 - y0)) if b > 0 else int(max(y0 - m, 0))
    X1 = int(x0 + c * (x1 - x0)) if c < 1 else int(x1 + m)
    Y1 = int(y0 + d * (y1 - y0)) if d < 1 else int(y1 + m)
    canvas = np.empty_like(img)
    canvas[:] = np.median(img.reshape(-1, 3), 0).astype(np.uint8)
    canvas[Y0:Y1, X0:X1] = img[Y0:Y1, X0:X1]
    # crop so the cut sides really are image borders (the detector's border guard depends on it)
    ox, oy = (X0 if a > 0 else 0), (Y0 if b > 0 else 0)
    return canvas[oy : Y1 if d < 1 else None, ox : X1 if c < 1 else None], np.array([ox, oy])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default=None, help="detector weights (default: configs/localizer.yaml)")
    a = ap.parse_args()
    if a.split == "test":
        sys.exit("Refusing to run on test: the test plates are scored only once, on purpose.")
    cfg, rng = load_config("localizer"), np.random.default_rng(a.seed)
    if a.weights:
        cfg["detector"]["weights"] = a.weights
    det = PlateDetector(cfg)
    ref_all = json.loads((PROCESSED_DIR / "plate_autolabels" / f"{a.split}.json").read_text())
    conf_frac = cfg["grid_fit"]["confident_frac"]

    per_clip, a1_pos, det_ms, snap_ms = {}, [], [], []
    hard = {"frames": 0, "box_found": 0, "box_err": [], "snap_ok": 0}  # frames the grid fit alone could not solve
    live = {"frames": 0, "with_plate": 0}
    cut = {k: {"detector": [], "track": []} for k in CUTS}
    track_ms = []
    for clip in clips_for_split(a.split, a.splits):
        frames = read_frames(clip, a.every)
        t0 = time.perf_counter()
        fit = localize_clip([f for _, f in frames], cfg, det)
        det_ms.append((time.perf_counter() - t0) * 1000 / len(frames))
        good = [x for x in ref_all.get(clip, []) if x["H"] and x["frac"] >= conf_frac]
        ref = max(good, key=lambda x: x["frac"]) if good else None
        rec = {"found": fit.found, "method": fit.method, "n_matched": fit.n_matched, "det_conf": round(fit.det_conf, 3)}
        if fit.found and ref is not None:
            rec["grid_err_pitch"] = round(grid_error(fit.H, np.array(ref["H"])), 4)
        if fit.found:
            a1_pos.append(project(fit.H, GRID[:1])[0])
            rec["pitch_px"] = round(pitch_px(fit.H), 2)
        per_clip[clip] = rec
        if ref is None:
            continue
        Href = np.array(ref["H"])
        # occluded frames: the grid fit alone was not confident there -> how good is the raw detector box?
        frac_by_frame = {x["frame"]: x["frac"] for x in ref_all.get(clip, [])}
        for i, f in frames:
            if frac_by_frame.get(i, 0.0) >= conf_frac:
                continue
            hard["frames"] += 1
            d = det.detect(f)
            if d is None:
                continue
            hard["box_found"] += 1
            hard["box_err"].append(grid_error(quad_to_H(d[0]), Href))
            s = snap_to_wells(f, quad_to_H(d[0]), cfg)
            hard["snap_ok"] += int(s.ok and grid_error(s.H, Href) < CORRECT)
        # live (causal) mode coverage over every 4th frame
        loc = PlateLocalizer(cfg, det)
        for _, f in frames:
            live["frames"] += 1
            live["with_plate"] += int(loc.update(f).found)
        # cut-plate tests on the reference frame
        img = dict(frames)[ref["frame"]] if ref["frame"] in dict(frames) else None
        if img is None:
            continue
        for name, keep in CUTS.items():
            canvas, off = cut_canvas(img, Href, keep)
            Hc = np.array([[1, 0, -off[0]], [0, 1, -off[1]], [0, 0, 1]]) @ Href
            d = det.detect(canvas)
            cut[name]["detector"].append("refused" if d is None else "accepted")
            t0 = time.perf_counter()
            t = track(canvas, perturb(Hc, rng), cfg)
            track_ms.append((time.perf_counter() - t0) * 1000)
            cut[name]["track"].append(None if not t.ok else grid_error(t.H, Hc))
        # snap latency on a full frame
        t0 = time.perf_counter()
        PlateLocalizer(cfg, det).detect(img)
        snap_ms.append((time.perf_counter() - t0) * 1000)

    errs = [r["grid_err_pitch"] for r in per_clip.values() if "grid_err_pitch" in r]
    a1 = np.array(a1_pos)
    a1_consistent = float(np.mean(np.linalg.norm(a1 - np.median(a1, 0), axis=1) < 2 * 14.5)) if len(a1) else 0.0

    def summ(v: list) -> dict:
        ok = [e for e in v if e is not None and e < CORRECT]
        return {
            "n": len(v),
            "correct": len(ok),
            "wrong": sum(e is not None and e >= CORRECT for e in v),
            "refused": v.count(None),
        }

    metrics = {
        "split": a.split,
        "clips": len(per_clip),
        "found_rate": float(np.mean([r["found"] for r in per_clip.values()])),
        "grid_err_pitch_mean": float(np.mean(errs)) if errs else None,
        "grid_err_pitch_max": float(np.max(errs)) if errs else None,
        "grid_correct_rate": float(np.mean([e < CORRECT for e in errs])) if errs else None,
        "a1_consistent_rate": a1_consistent,
        "occluded_frames": {
            "n": hard["frames"],
            "detector_found_rate": hard["box_found"] / max(hard["frames"], 1),
            "box_grid_err_median": float(np.median(hard["box_err"])) if hard["box_err"] else None,
            "box_correct_rate": float(np.mean(np.array(hard["box_err"]) < CORRECT)) if hard["box_err"] else None,
            "normal_snap_correct_rate": hard["snap_ok"] / max(hard["frames"], 1),
        },
        "live_mode_frames_with_plate": live["with_plate"] / max(live["frames"], 1),
        "weights": cfg["detector"]["weights"],
        "cut_plate": {
            k: {
                "detector_refused": v["detector"].count("refused"),
                "detector_accepted": v["detector"].count("accepted"),
                "track": summ(v["track"]),
            }
            for k, v in cut.items()
        },
        "latency_ms": {
            "detect_per_frame_incl_snap": float(np.mean(det_ms)),
            "detect_full_frame": float(np.mean(snap_ms)) if snap_ms else None,
            "track_per_frame": float(np.mean(track_ms)) if track_ms else None,
        },
        "per_clip": per_clip,
    }
    run = new_run_dir(f"localizer-{a.split}")
    write_metrics(run, metrics)
    lines = [f"# Plate localizer eval ({a.split})", ""]
    lines += [
        f"- Plate found: {metrics['found_rate']:.0%} of {len(per_clip)} clips",
        f"- Grid error vs auto-label reference: mean {metrics['grid_err_pitch_mean']:.3f}, "
        f"max {metrics['grid_err_pitch_max']:.3f} pitch; correct (<{CORRECT}) {metrics['grid_correct_rate']:.0%}",
        f"- A1 lands in the same image spot across clips: {a1_consistent:.0%}",
        f"- Live mode: plate shown in {metrics['live_mode_frames_with_plate']:.0%} of frames (every 4th frame)",
        f"- Occluded frames (grid fit alone not confident, n={hard['frames']}): detector found "
        f"{metrics['occluded_frames']['detector_found_rate']:.0%}, raw box grid error median "
        f"{metrics['occluded_frames']['box_grid_err_median'] or float('nan'):.2f} pitch, box correct "
        f"{metrics['occluded_frames']['box_correct_rate'] or 0:.0%}, normal snap correct "
        f"{metrics['occluded_frames']['normal_snap_correct_rate']:.0%}",
        f"- Weights: {cfg['detector']['weights']}",
        "",
        "| cut scenario | detector refused / accepted | tracking correct / wrong / refused |",
        "|---|---|---|",
    ]
    for k, v in metrics["cut_plate"].items():
        t = v["track"]
        det_s = f"{v['detector_refused']} / {v['detector_accepted']}"
        lines.append(f"| {k} | {det_s} | {t['correct']} / {t['wrong']} / {t['refused']} |")
    lat = metrics["latency_ms"]
    lines += [
        "",
        f"Latency: detect+snap {lat['detect_full_frame']:.0f} ms/frame (only while searching), "
        f"track {lat['track_per_frame']:.0f} ms/frame.",
        "",
        "Caveat: the reference grid is the confident global grid fit (checked by eye during exploration),",
        "not human labels.",
    ]
    (run / "notes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("run:", run)


if __name__ == "__main__":
    main()
