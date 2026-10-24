"""Well views per press clip for the before/after liquid model: every well as it looks at each press, and whether it
was covered (pipette, hand) then.

From scripts/build_liquid_dataset.py's warped plates (data/processed/liquid_ds/<rec>, grid-corrected, S px per well):
view of clip j = pixel median over the clip's frames (~2 s around the press). State of well w in clip j (fill order,
one press per well): empty if j < w's press, filled if j > w's press, being filled if j == w's press.
Covered: the well cell is far from its own usual look in the same state (median over every clip in that state; the
pipette lies over any one well in only a few clips, so the usual look is clear). The corner test of well_change.py
and a ring-pattern test both missed the see-through yellow tip / could not tell liquid from pipette (checked by eye).

Output data/processed/well_views/<rec>/: plates.npy (n_clips, 8*S, 12*S, 3) uint8, dev.npy (n_clips, 8, 12)
mean abs difference from the usual look (NaN: the clip's own well, or < 3 clips in that state), meta.json.

    python scripts/build_well_views.py --rec pipette8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.liquid_net import S  # noqa: E402

COVER_DEV = 10.0  # mean abs BGR difference from the usual look that counts as covered (clear wells: 2-8, pipette 13+)


def press_of(c: int, r: int) -> int:
    """Press index of grid well (col, row) in the fill order (inverse of liquid_net.press_well)."""
    return (11 - c) * 8 + (7 - r)


def deviation(views: np.ndarray, clips: list[int], s: int = S) -> np.ndarray:
    """views (n, 8s, 12s, 3) -> (n, 8, 12) mean abs difference of each cell from its usual look in the same state."""
    cells = views.astype(np.float32).reshape(len(clips), 8, s, 12, s, 3).transpose(0, 1, 3, 2, 4, 5)
    K = np.array(clips)
    dev = np.full((len(clips), 8, 12), np.nan, np.float32)
    for r in range(8):
        for c in range(12):
            k0 = press_of(c, r)
            for state in (K < k0, K > k0):
                if state.sum() >= 3:
                    ref = np.median(cells[state, r, c], 0)
                    dev[state, r, c] = np.abs(cells[state, r, c] - ref).mean((1, 2, 3))
    return dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="pipette8")
    a = ap.parse_args()
    src = PROCESSED_DIR / "liquid_ds" / a.rec
    meta = json.loads((src / "meta.json").read_text())
    P = np.load(src / "plates.npy", mmap_mode="r")
    clips = sorted({s["k"] for s in meta["samples"]})
    views = np.stack([np.median(np.array(P[[s["idx"] for s in meta["samples"] if s["k"] == k]]), 0)
                      for k in clips]).astype(np.uint8)  # fmt: skip
    dev = deviation(views, clips)
    out = PROCESSED_DIR / "well_views" / a.rec
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "plates.npy", views)
    np.save(out / "dev.npy", dev)
    covered = float(np.nanmean(dev > COVER_DEV))
    (out / "meta.json").write_text(json.dumps({"rec": a.rec, "clips": clips, "S": S, "cover_dev": COVER_DEV,
                                               "covered_share": round(covered, 4)}))  # fmt: skip
    print(f"{len(clips)} clips, {covered:.1%} of well views covered ->", out)


if __name__ == "__main__":
    main()
