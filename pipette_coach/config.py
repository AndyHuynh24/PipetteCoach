"""Config loading. All CV thresholds live in configs/*.yaml."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
# Best tip-end model so far (default of every script that runs the tip model). Change here to switch.
# 2026-10-24: back to tip_p5 (the setup that worked live: rule + tip_p5 + old well_net); tip_colors was used briefly.
TIP_MODEL = "models/tip_p5/weights/best.pt"
# Well grid model of the live tracker: the upload of 2026-10-24 (best_old.pt = Oct 13 version).
WELL_NET = "models/well_net/best.pt"
# Liquid going into the well under the tip (one-well model, scripts/train_liquid_crop.py). 2026-10-24: pipette8.
LIQUID_MODEL = "models/liquid_crop/best.pt"
# Before/after liquid model (scripts/train_well_compare.py), used by live_demo --liquid compare. 2026-10-24.
COMPARE_MODEL = "models/well_compare/best.pt"
# Same model, k-fold trained on all pipette8 presses (scripts/train_compare_kfold.py): live_demo --liquid compare-cv.
COMPARE_CV_MODEL = "models/well_compare_cv/final.pt"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml as a dict."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)
