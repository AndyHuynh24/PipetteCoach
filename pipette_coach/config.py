"""Config loading. All CV thresholds live in configs/*.yaml."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml as a dict."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)
