"""Tip dataset helpers: dispense labels as well indices, plate-view clip data (scripts/build_tip_data.py)."""

import numpy as np

from pipette_coach.data import PROCESSED_DIR, clips_for_split, load_labels, plate_of
from pipette_coach.protocol.wells import well_index

TIP_DIR = PROCESSED_DIR / "tip_data"


def clip_wells() -> dict[str, list[int]]:
    """Clip -> dispensed well indices (row-major 0..95) from labels.json."""
    out = {}
    for x in load_labels():
        clip = x["clip_id_Topview"].replace("_Topview", "")
        out[clip] = sorted(
            well_index({"row": w["well_row"], "col": int(w["well_column"])}) for w in x["wells_ground_truth"]
        )
    return out


def split_clips(split: str, plates: set[int] | None = None) -> list[str]:
    """Clips of a split that have plate-view data with a grid in at least one frame."""
    out = []
    for c in clips_for_split(split):
        if plates is not None and plate_of(c) not in plates:
            continue
        f = TIP_DIR / f"{c}.npz"
        if f.exists() and np.load(f)["has_grid"].any():
            out.append(c)
    return out


def load_clip(clip: str) -> dict:
    d = np.load(TIP_DIR / f"{clip}.npz")
    return {k: d[k] for k in d.files}
