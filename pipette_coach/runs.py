"""Experiment output folders: runs/<timestamp>_<short-name>/ with metrics.json and notes.md."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from pipette_coach.config import REPO_ROOT


def new_run_dir(short_name: str) -> Path:
    d = REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{short_name}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def git_rev() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "uncommitted"


def write_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "metrics.json").write_text(json.dumps({"git": git_rev(), **metrics}, indent=2))
