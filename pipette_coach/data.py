"""Read-only access to the Transfyr dataset in data/raw/ and the plate-level split."""

import json
import re
from pathlib import Path

from pipette_coach.config import CONFIG_DIR, REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_SPLITS = CONFIG_DIR / "splits.json"


def load_labels() -> list[dict]:
    """labels.json entries: clip_id_Topview, clip_id_FPV, wells_ground_truth."""
    with open(RAW_DIR / "labels.json") as f:
        return json.load(f)


def clip_base(clip_id: str) -> str:
    """'Plate_4_clip_0017_Topview' -> 'Plate_4_clip_0017'."""
    return re.sub(r"_(Topview|FPV)$", "", clip_id)


def plate_of(clip_id: str) -> int:
    m = re.match(r"Plate_(\d+)_clip_\d+", clip_id)
    if m is None:
        raise ValueError(f"not a clip id: {clip_id}")
    return int(m.group(1))


def clip_path(clip: str, view: str = "Topview") -> Path:
    return RAW_DIR / f"{clip_base(clip)}_{view}.mp4"


def load_splits(path: Path | str = DEFAULT_SPLITS) -> dict[str, list[int]]:
    """Plate ids per split. The split is by plate, never by clip."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (the plate split is defined in configs/splits.json).")
    with open(path) as f:
        s = json.load(f)
    excluded = set(excluded_plates())
    return {k: [p for p in s[k] if p not in excluded] for k in ("train", "val", "test")}


def excluded_clips() -> list[str]:
    """Single clips left out of training and evaluation (configs/excluded_plates.json "excluded_clips")."""
    p = DEFAULT_SPLITS.parent / "excluded_plates.json"
    return json.loads(p.read_text()).get("excluded_clips", []) if p.exists() else []


def excluded_plates() -> list[int]:
    """Plates left out of training and evaluation (configs/excluded_plates.json; user decision, e.g. plate 10)."""
    p = DEFAULT_SPLITS.parent / "excluded_plates.json"
    return json.loads(p.read_text())["excluded"] if p.exists() else []


def clips_for_split(split: str, splits_path: Path | str = DEFAULT_SPLITS) -> list[str]:
    """Clip bases (e.g. 'Plate_4_clip_0017') whose plate is in the given split, sorted."""
    plates = set(load_splits(splits_path)[split])
    clips = [clip_base(e["clip_id_Topview"]) for e in load_labels()]
    skip = set(excluded_clips())
    return sorted(c for c in clips if plate_of(c) in plates and c not in skip)


def plate_rows_toward(plate: int) -> str | None:
    """Image direction of rows A -> H for a dataset plate (configs/plate_orientation.json), or None."""
    with open(CONFIG_DIR / "plate_orientation.json") as f:
        return json.load(f).get(str(plate))
