"""Live-mode evaluation over ALL clips (frame by frame, as the app runs), plus the statistics used to set the
edge-anchor thresholds - so thresholds come from distributions over many clips, not from one example.

References (no dataset well labels are used):
  val   : the confident global grid fit of each clip (data/processed/plate_autolabels/val.json) - an
          independent method from the live pipeline.
  train : the per-clip grid in data/processed/clip_grids.json, only for clips that are unflagged or
          eye-verified (configs/verified_grids.json). Plates barely move within a clip; a real move shows up as
          error here, so train numbers are a lower bound.

Writes runs/<ts>_live-eval/{metrics.json, notes.md}.

    python scripts/eval_live.py --splits-to-run train,val
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clips_for_split, plate_of, plate_rows_toward  # noqa: E402
from pipette_coach.localizer import PlateLocalizer  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import GRID, grid_error, pitch_px, project  # noqa: E402
from pipette_coach.localizer.refine import track  # noqa: E402
from pipette_coach.localizer.session import read_frames, session_prior  # noqa: E402
from pipette_coach.localizer.verifier import GridVerifier  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

CORRECT = 0.25  # pitch: a grid this close cannot confuse neighbouring wells


def references(split: str) -> dict[str, np.ndarray]:
    if split == "val":
        val = json.loads((PROCESSED_DIR / "plate_autolabels" / "val.json").read_text())
        out = {}
        for c, fr in val.items():
            good = [x for x in fr if x["H"] and x["frac"] >= 0.9]
            if good:
                out[c] = np.array(max(good, key=lambda x: x["frac"])["H"])
        return out
    grids = json.loads((PROCESSED_DIR / "clip_grids.json").read_text())
    verified = set(json.loads((REPO_ROOT / "configs" / "verified_grids.json").read_text())) - {"_why"}
    return {c: np.array(g["H"]) for c, g in grids.items() if g["H"] is not None and (not g["flags"] or c in verified)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-to-run", default="train,val")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--no-verifier", action="store_true")
    a = ap.parse_args()
    cfg = load_config("localizer")
    det = PlateDetector(cfg)
    ver = None if a.no_verifier else GridVerifier(REPO_ROOT / cfg["verifier"]["weights"])
    metrics, stats = {}, {"ncc": [], "still_border_shift_px": [], "still_wells_shift_px": []}
    for split in a.splits_to_run.split(","):
        if split == "test":
            sys.exit("Refusing to run on test without explicit approval.")
        refs = references(split)
        clips = clips_for_split(split, a.splits)
        priors = {}
        per = []
        for clip in clips:
            pl = plate_of(clip)
            if pl not in priors:
                priors[pl] = session_prior([c for c in clips if plate_of(c) == pl], cfg, det)
            frames = read_frames(clip)
            loc = PlateLocalizer(cfg, det, priors[pl], plate_rows_toward(pl), ver)
            fits = []
            for f in frames:
                if loc.anchored and loc.H is not None:  # threshold statistics (before the step changes state)
                    m = loc.anchor.motion(f)
                    stats["ncc"].append(m.response)
                    if m.response >= cfg["edge_anchor"]["min_response"]:
                        stats["still_border_shift_px"].append(float(np.hypot(m.dx, m.dy)))
                    t = track(f, loc.H, cfg)
                    if t.ok and t.n_matched >= cfg["edge_anchor"]["wells_min_matches"]:
                        d = np.linalg.norm(project(t.H, GRID) - project(loc.H, GRID), axis=1).mean()
                        stats["still_wells_shift_px"].append(float(d))
                fits.append(loc.update(f))
            found = [f for f in fits if f.found]
            steps = [
                float(np.linalg.norm(project(b.H, GRID) - project(a_.H, GRID), axis=1).mean())
                for a_, b in zip(fits, fits[1:], strict=False)
                if a_.found and b.found
            ]
            rec = {"clip": clip, "found": len(found) / len(fits), "steps_gt_1px": int(sum(s > 1.0 for s in steps))}
            if clip in refs and found:
                errs = [grid_error(f.H, refs[clip]) for f in found]  # geometry; A1 checked separately
                rec["correct"] = float(np.mean(np.array(errs) < CORRECT))
                rec["err_px_median"] = float(np.median(errs) * pitch_px(refs[clip]))
            per.append(rec)
            print(
                clip,
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k != "clip"},
                flush=True,
            )
        with_ref = [r for r in per if "correct" in r]
        metrics[split] = {
            "clips": len(per),
            "clips_with_reference": len(with_ref),
            "found_rate": float(np.mean([r["found"] for r in per])),
            "frames_correct_rate": float(np.mean([r["correct"] for r in with_ref])) if with_ref else None,
            "clips_all_frames_correct": int(sum(r["correct"] == 1.0 for r in with_ref)),
            "err_px_median": float(np.median([r["err_px_median"] for r in with_ref])) if with_ref else None,
            "grid_steps_gt_1px_per_clip": float(np.mean([r["steps_gt_1px"] for r in per])),
            "worst": sorted(with_ref, key=lambda r: r["correct"])[:5],
        }
    pct = {k: (np.percentile(v, [50, 90, 99]).round(3).tolist() if v else None) for k, v in stats.items()}
    metrics["threshold_stats_p50_p90_p99"] = pct
    run = new_run_dir("live-eval")
    write_metrics(run, metrics)
    lines = ["# Live-mode evaluation (frame by frame, all clips)", ""]
    for split in a.splits_to_run.split(","):
        m = metrics[split]
        lines.append(
            f"- {split}: {m['clips']} clips, found {m['found_rate']:.0%}; with reference {m['clips_with_reference']}: "
            f"frames correct {m['frames_correct_rate'] or 0:.1%}, clips all-correct {m['clips_all_frames_correct']}, "
            f"median error {m['err_px_median'] or 0:.2f} px, "
            f"grid steps >1 px per clip {m['grid_steps_gt_1px_per_clip']:.1f}"
        )
    lines += ["", f"Threshold statistics (p50/p90/p99): {json.dumps(pct)}"]
    (run / "notes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("run:", run)


if __name__ == "__main__":
    main()
