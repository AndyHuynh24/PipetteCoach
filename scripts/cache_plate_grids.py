"""Cache plate box + 12 x 8 grid per static window for every train/val clip (localizer/plate_grid.py).

Output: data/processed/plate_grids/<clip>.json  {windows: [{start, end, box, sides, H_grid}]}  (regenerable)
Test plates are refused.

    python scripts/cache_plate_grids.py [--plates 4,1,2,9,10]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clips_for_split, load_splits, plate_of, plate_rows_toward  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.plate_grid import clip_grid_windows  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402

OUT = PROCESSED_DIR / "plate_grids"


def load_grid_windows(clip: str) -> list[dict]:
    import numpy as np

    d = json.loads((OUT / f"{clip}.json").read_text())
    return [{**w, "box": np.array(w["box"]), "H_grid": None if w["H_grid"] is None else np.array(w["H_grid"])}
            for w in d["windows"]]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="4,1,2,9,10")
    a = ap.parse_args()
    cfg = load_config("localizer")
    splits = load_splits()
    plates = [int(x) for x in a.plates.split(",") if x]
    if any(p in splits["test"] for p in plates):
        sys.exit("a plate is in the test split; refusing.")
    det = PlateDetector(cfg)
    net = WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    size = tuple(cfg["plate_edges"]["plate_size_pitch"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for pl in plates:
        for clip in [c for s in ("train", "val") for c in clips_for_split(s) if plate_of(c) == pl]:
            wins = clip_grid_windows(read_frames(clip), det, net, cfg, size, plate_rows_toward(pl))
            js = [{"start": w["start"], "end": w["end"], "box": w["box"].tolist(), "sides": w["sides"],
                   "H_grid": None if w["H_grid"] is None else w["H_grid"].tolist()} for w in wins]  # fmt: skip
            (OUT / f"{clip}.json").write_text(json.dumps({"clip": clip, "windows": js}))
            ok = "ok" if all(w["H_grid"] is not None for w in wins) else "MISSING"
            print(f"{clip}: {len(wins)} window(s), grid {ok} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
