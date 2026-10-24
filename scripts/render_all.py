"""Render every train/val clip with the current best plate/well method, and flagged clips in a separate video.

Per clip: session-mode pipeline fits (remembered orientation, per-plate A1) + temporal fusion of well evidence.
Grid shown: the fused static grid when it agrees with the per-frame pipeline (< agree_max pitch), otherwise the
per-frame pipeline. A clip is FLAGGED (needs review) when:
  - fusion failed / too few wells fused
  - fused and per-frame grids disagree on placement (>= agree_max pitch)
  - the plate moved within the clip (fusion motion check)
  - the plate is missing in some frames
  - neighbour mismatch: exactly one row/column off from the previous/next clip of the same plate while the plate
    did not move (catches a hidden outer row resolved the wrong way)
No dataset labels are used. Test plates are refused. Output: runs/<ts>_render-all/{all.mp4, flagged.mp4, flags.json}

    python scripts/render_all.py --plates 4,1,2,9,10
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_tracking import OUT_H, OUT_W, compose, title_card  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import clips_for_split, load_splits, plate_of, plate_rows_toward  # noqa: E402
from pipette_coach.localizer import PlateFit, localize_clip_frames  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.fusion import fuse_clip  # noqa: E402
from pipette_coach.localizer.geometry import OUTLINE, grid_error, project  # noqa: E402
from pipette_coach.localizer.session import read_frames, session_prior  # noqa: E402
from pipette_coach.localizer.verifier import OFFSETS, GridVerifier, offset_H  # noqa: E402
from pipette_coach.runs import new_run_dir  # noqa: E402

AGREE_MAX = 0.3  # pitch: fused vs per-frame placement must agree this well, else the clip is flagged
MIN_FUSED_WELLS = 40
NEIGHBOUR_SAME_POS = 0.12  # pitch: after a 1-row/col shift the neighbour grid matches this well -> plate static
NEIGHBOUR_ASIS_MIN = 0.8  # ...while as-is they differ by at least this much


def shifted(H: np.ndarray, dc: int, dr: int) -> np.ndarray:
    return H @ np.array([[1, 0, dc], [0, 1, dr], [0, 0, 1]], np.float64)


def verify_clip(frames: list, H: np.ndarray, cfg: dict, ver) -> tuple[np.ndarray, str | None]:
    """Learned verifier vote over the clip: apply a half-well/row/column correction only when confident."""
    import collections

    vc = cfg["verifier"]
    votes, margins = collections.Counter(), []
    for f in frames[:: vc["every"]]:
        s = ver.scores(f, [offset_H(H, dx, dy) for dx, dy in OFFSETS])
        order = np.argsort(-s)
        votes[OFFSETS[order[0]]] += 1
        margins.append(float(s[order[0]] - s[order[1]]))
    (best, n), total = votes.most_common(1)[0], sum(votes.values())
    confident = n / total >= vc["min_vote"] and float(np.median(margins)) >= vc["min_margin"]
    if best == (0, 0):
        return H, None if confident else f"verifier uncertain (margin {np.median(margins):.2f})"
    if confident:
        return offset_H(H, *best), f"corrected by verifier: shift {best} ({n}/{total} frames)"
    return H, f"verifier uncertain: {n}/{total} frames suggest {best}, margin {np.median(margins):.2f}"


def process_clip(clip: str, cfg: dict, det: PlateDetector, prior: np.ndarray | None, ver=None) -> dict:
    frames = read_frames(clip)
    pl = plate_of(clip)
    fits = localize_clip_frames(frames, cfg, det, prior_H=prior, rows_toward=plate_rows_toward(pl), verifier=ver)
    found = [f for f in fits if f.found]
    flags = []
    if len(found) < len(fits):
        flags.append(f"plate missing in {len(fits) - len(found)}/{len(fits)} frames")
    if not found:
        return {"clip": clip, "fits": fits, "H_static": None, "flags": flags or ["plate not found"]}
    quads = [(d[0] if (d := det.detect(f)) is not None else None) for f in frames]
    fp = fuse_clip(frames, quads, [f.H for f in found], cfg)
    H_static = None
    if fp is None or fp.n_wells < MIN_FUSED_WELLS:
        flags.append("fusion failed / too few wells seen")
    else:
        dis = float(np.median([grid_error(f.H, fp.H, ignore_labelling=False) for f in found]))
        if dis >= AGREE_MAX:
            flags.append(f"fused vs per-frame grids disagree ({dis:.2f} pitch)")
        else:
            H_static = fp.H
        if fp.moved_frames:
            flags.append(f"plate moved in {len(fp.moved_frames)} frames")
    H_ref = H_static if H_static is not None else found[len(found) // 2].H
    note = None
    if ver is not None:
        H_new, note = verify_clip(frames, H_ref, cfg, ver)
        if note and note.startswith("corrected"):
            H_static = H_new  # a confident correction applies to the whole (static) clip
            H_ref = H_new
            flags = [f for f in flags if "disagree" not in f]
        elif note:
            flags.append(note)
    return {
        "clip": clip,
        "fits": fits,
        "note": note,
        "H_static": H_static,
        "H_ref": H_ref,
        "fused": None if fp is None else {"wells": fp.n_wells, "rms": fp.rms, "loo": fp.loo},
        "flags": flags,
    }


def neighbour_flags(results: list[dict]) -> None:
    """Flag consecutive clips of a plate whose grids are exactly one row/column apart while the plate is static."""
    for a, b in zip(results, results[1:], strict=False):
        if a.get("H_ref") is None or b.get("H_ref") is None:
            continue
        as_is = grid_error(a["H_ref"], b["H_ref"], ignore_labelling=False)
        best = min(
            grid_error(shifted(a["H_ref"], dc, dr), b["H_ref"], ignore_labelling=False)
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        if as_is >= NEIGHBOUR_ASIS_MIN and best <= NEIGHBOUR_SAME_POS:
            msg = f"one row/column off vs neighbour ({a['clip'][-9:]} / {b['clip'][-9:]})"
            a["flags"].append(msg)
            b["flags"].append(msg)


def write_clip(vw: cv2.VideoWriter, r: dict, fps: float, mark_flagged: bool) -> None:
    lines = [r["clip"]] + (["FLAGGED: " + f for f in r["flags"]] if r["flags"] else ["OK: no flags"])
    if r.get("note") and r["note"].startswith("corrected"):
        lines.append(r["note"])
    for _ in range(int(fps)):
        vw.write(title_card(lines))
    center, zoom = None, 2.5
    for i, (frame, fit) in enumerate(zip(read_frames(r["clip"]), r["fits"], strict=True)):  # re-read: low memory
        shown = PlateFit(r["H_static"], "tracked", "fused (static)", 0) if r["H_static"] is not None else fit
        if shown.found:
            center = project(shown.H, np.array([[5.5, 3.5]]))[0]
            ext = np.ptp(project(shown.H, OUTLINE), axis=0)
            zoom = 0.8 * min(560 / ext[0], 380 / ext[1])
        title = r["clip"] + ("   [FLAGGED]" if mark_flagged and r["flags"] else "")
        vw.write(compose(frame, shown, title, i, 0.0, center, zoom))


def encode(raw: Path, out: Path) -> None:
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True)
        raw.unlink()
    else:
        raw.rename(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="4,1,2,9,10")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--from-flags", default=None, help="flags.json of an earlier run: render only its flagged clips")
    ap.add_argument("--name", default="render-all")
    ap.add_argument("--clips", default="", help="comma-separated clips to render (in addition to --from-flags)")
    a = ap.parse_args()
    only = None
    if a.from_flags:
        prev = json.loads(Path(a.from_flags).read_text())
        only = {
            r["clip"]
            for r in prev
            if r["flags"]
            and not any(
                "plate missing in" in f and f.split()[3].split("/")[0] == f.split()[3].split("/")[1] for f in r["flags"]
            )
        }
    if a.clips:
        only = (only or set()) | {c for c in a.clips.split(",") if c}
    cfg = load_config("localizer")
    det = PlateDetector(cfg)
    ver = GridVerifier(REPO_ROOT / cfg["verifier"]["weights"])
    splits = load_splits(a.splits)
    run = new_run_dir(a.name)
    vw_all = cv2.VideoWriter(str(run / "all_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (OUT_W, OUT_H))
    vw_flag = cv2.VideoWriter(str(run / "flagged_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (OUT_W, OUT_H))
    summary = []
    for pl in [int(x) for x in a.plates.split(",") if x]:
        if pl in splits["test"]:
            sys.exit(f"plate {pl} is in the test split; not rendering test data without explicit approval.")
        clips = [c for s in ("train", "val") for c in clips_for_split(s, a.splits) if plate_of(c) == pl]
        prior = session_prior(clips, cfg, det)
        todo = [c for c in clips if only is None or c in only]
        if not todo:
            continue
        results = [process_clip(c, cfg, det, prior, ver) for c in todo]
        neighbour_flags(results)
        for r in results:
            write_clip(vw_all, r, a.fps, mark_flagged=True)
            if r["flags"] and only is None:
                write_clip(vw_flag, r, a.fps, mark_flagged=True)
            summary.append({"clip": r["clip"], "flags": r["flags"], "fused": r.get("fused"), "note": r.get("note")})
            print(f"{r['clip']}: {'; '.join(r['flags']) if r['flags'] else 'OK'}", flush=True)
    vw_all.release()
    vw_flag.release()
    encode(run / "all_raw.mp4", run / "all.mp4")
    if only is None:
        encode(run / "flagged_raw.mp4", run / "flagged.mp4")
    else:
        (run / "flagged_raw.mp4").unlink(missing_ok=True)  # flagged-only run: everything is in all.mp4
    (run / "flags.json").write_text(json.dumps(summary, indent=2))
    n_flag = sum(bool(s["flags"]) for s in summary)
    (run / "notes.md").write_text(
        f"# All clips render\n\n{len(summary)} clips, {n_flag} flagged (see flags.json). Grid = fused static grid "
        "when it agrees with the per-frame pipeline, else per-frame. No dataset labels used. Contains dataset "
        "frames: keep local.\n"
    )
    print(f"{len(summary)} clips, {n_flag} flagged -> {run}")


if __name__ == "__main__":
    main()
