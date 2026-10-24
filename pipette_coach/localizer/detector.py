"""YOLO oriented-box plate detector (trained on auto-labelled frames; see scripts/train_plate_detector.py)."""

from pathlib import Path

import numpy as np

from pipette_coach.config import REPO_ROOT


class PlateDetector:
    """Wraps an ultralytics OBB model. Returns the best plate quad, or None.

    Boxes touching the image border are rejected: a cut-off plate yields a shrunken box whose grid
    numbering is shifted (0/120 correct in the cut-plate test), so it must not be trusted.
    """

    def __init__(self, cfg: dict, device: str | None = None):
        from ultralytics import YOLO  # imported lazily so the rest of the package works without torch

        dc = cfg["detector"]
        weights = Path(dc["weights"])
        weights = weights if weights.is_absolute() else REPO_ROOT / weights
        if not weights.exists():
            raise FileNotFoundError(f"{weights} missing; run scripts/train_plate_detector.py")
        self.model = YOLO(str(weights))
        self.imgsz, self.min_conf, self.margin = dc["imgsz"], dc["min_conf"], dc["border_margin_px"]
        self.device = device or _default_device()

    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.min_conf, verbose=False, device=self.device)[0]
        quads, confs = boxes_of(r)
        if len(confs) == 0:
            return None
        k = int(np.argmax(confs))
        quad = quads[k]
        h, w = bgr.shape[:2]
        m = self.margin
        if (
            (quad[:, 0] < m).any()
            or (quad[:, 1] < m).any()
            or (quad[:, 0] > w - 1 - m).any()
            or (quad[:, 1] > h - 1 - m).any()
        ):
            return None
        return quad, float(confs[k])


def boxes_of(r) -> tuple[np.ndarray, np.ndarray]:
    """(quads (n, 4, 2), confs (n,)) from a YOLO result: oriented boxes (OBB model) or axis-aligned boxes
    (detect model, e.g. models/plate_det) as 4-corner quads."""
    if getattr(r, "obb", None) is not None and len(r.obb):
        return r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2), r.obb.conf.cpu().numpy()
    if getattr(r, "boxes", None) is not None and len(r.boxes):
        b = r.boxes.xyxy.cpu().numpy()
        q = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [2, 3]], b[:, [0, 3]]], axis=1)
        return q, r.boxes.conf.cpu().numpy()
    return np.zeros((0, 4, 2)), np.zeros(0)


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
