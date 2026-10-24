"""Auto-label plate grids on dataset frames (no manual marking).

--method grid (v1): global grid fit per frame; only fits with frac >= grid_fit.confident_frac are used.
    -> data/processed/plate_autolabels/<split>.json: {clip: [{"frame", "H" (or null), "frac"}]}
--method session (v2): full localizer (detector + snap/partial snap + session orientation + tracking/hold),
    which also labels frames where the plate is partly hidden; clips in configs/autolabel_exclude.json are
    skipped. -> data/processed/plate_autolabels/<split>_session.json (frac = 1 for every found frame)

    python scripts/autolabel_plates.py --split train
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clip_path, clips_for_split  # noqa: E402
from pipette_coach.localizer.grid_fit import fit_grid  # noqa: E402

CFG = load_config("localizer")


def label_clip(args: tuple[str, int]) -> tuple[str, list[dict]]:
    clip, every = args
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(clip_path(clip)))
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            g = fit_grid(frame, CFG, seed=i)
            out.append(
                {"frame": i, "H": None if g is None else g.H.tolist(), "frac": 0.0 if g is None else g.frac_on_rings}
            )
        i += 1
    return clip, out


def label_session(clips: list[str], every: int) -> dict[str, list[dict]]:
    """Session-mode labels using the current detector (sequential: the detector runs on the GPU)."""
    from pipette_coach.data import plate_of
    from pipette_coach.localizer import localize_clip_frames
    from pipette_coach.localizer.detector import PlateDetector
    from pipette_coach.localizer.session import read_frames, session_prior

    det = PlateDetector(CFG)
    exclude = set(json.loads((REPO_ROOT / "configs" / "autolabel_exclude.json").read_text())) - {"_why"}
    out: dict[str, list[dict]] = {}
    for plate in sorted({plate_of(c) for c in clips}):
        same = [c for c in clips if plate_of(c) == plate]
        prior = session_prior(same, CFG, det)
        for clip in same:
            if clip in exclude:
                continue
            fits = localize_clip_frames(read_frames(clip), CFG, det, prior_H=prior)
            out[clip] = [
                {
                    "frame": i,
                    "H": f.H.tolist() if f.found else None,
                    "frac": 1.0 if f.found else 0.0,
                    "status": f.status,
                }
                for i, f in enumerate(fits)
                if i % every == 0
            ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"])  # never auto-label test
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--every", type=int, default=4, help="label every Nth frame")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--method", choices=["grid", "session"], default="grid")
    a = ap.parse_args()
    clips = clips_for_split(a.split, a.splits)
    if a.method == "session":
        res = label_session(clips, a.every)
    else:
        with Pool(a.workers) as p:
            res = dict(p.map(label_clip, [(c, a.every) for c in clips]))
    out = PROCESSED_DIR / "plate_autolabels" / f"{a.split}{'_session' if a.method == 'session' else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    n = sum(len(v) for v in res.values())
    conf = sum(f["frac"] >= CFG["grid_fit"]["confident_frac"] for v in res.values() for f in v)
    print(f"{a.split}: {len(res)} clips, {n} frames, {conf} confident -> {out}")


if __name__ == "__main__":
    main()
