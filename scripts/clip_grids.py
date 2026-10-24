"""Compute each train/val clip's plate grid + review flags once (same logic as render_all) and save them.

Output: data/processed/clip_grids.json  {clip: {"H": 3x3 or null, "static": bool, "flags": [...]}}

    python scripts/clip_grids.py --plates 1,2,9,10,4
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_all import neighbour_flags, process_clip  # noqa: E402

from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clips_for_split, load_splits, plate_of  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.session import session_prior  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="1,2,9,10,4")
    ap.add_argument("--splits", default="configs/splits.json")
    a = ap.parse_args()
    cfg = load_config("localizer")
    det = PlateDetector(cfg)
    splits = load_splits(a.splits)
    out = {}
    for pl in [int(x) for x in a.plates.split(",") if x]:
        if pl in splits["test"]:
            sys.exit(f"plate {pl} is in the test split")
        clips = [c for s in ("train", "val") for c in clips_for_split(s, a.splits) if plate_of(c) == pl]
        prior = session_prior(clips, cfg, det)
        results = [process_clip(c, cfg, det, prior) for c in clips]
        neighbour_flags(results)
        for r in results:
            H = r.get("H_ref")
            out[r["clip"]] = {
                "H": None if H is None else H.tolist(),
                "static": r["H_static"] is not None,
                "flags": r["flags"],
            }
            print(r["clip"], r["flags"] or "OK", flush=True)
    path = PROCESSED_DIR / "clip_grids.json"
    path.write_text(json.dumps(out))
    print("->", path)


if __name__ == "__main__":
    main()
