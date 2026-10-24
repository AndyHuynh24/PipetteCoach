"""Plate segmentation (YOLO11n-seg) as a drop-in for the box detector: same detect(bgr) -> (quad, conf).

The mask says which pixels are plate, so an upright plate is one region (a box detector trained on sideways
plates split it in two). The quad is a 4-sided fit to the mask's outline (minimum-area rectangle of the largest
contour); the plate's real edges then snap it exactly (plate_edges). The frame is searched at several scales so
any camera distance works.
"""

from pathlib import Path

import cv2
import numpy as np

from pipette_coach.config import REPO_ROOT
from pipette_coach.localizer.detector import _default_device


class PlateSegmenter:
    def __init__(self, cfg: dict, weights: str | Path | None = None, device: str | None = None):
        from ultralytics import YOLO

        sc = cfg.get("segmenter", {})
        w = Path(weights or sc.get("weights", "models/plate_seg/weights/best.pt"))
        w = w if w.is_absolute() else REPO_ROOT / w
        if not w.exists():
            raise FileNotFoundError(f"{w} missing; run scripts/train_plate_seg.py")
        self.model = YOLO(str(w))
        dc = cfg["detector"]
        self.imgsz, self.min_conf, self.margin = dc["imgsz"], dc["min_conf"], dc["border_margin_px"]
        self.device = device or _default_device()

    def mask(self, bgr: np.ndarray, conf: float | None = None) -> tuple[np.ndarray, float] | None:
        """Best plate mask (H x W uint8, original size) and its confidence."""
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=conf or self.min_conf, verbose=False,
                               device=self.device, retina_masks=True)[0]  # fmt: skip
        if r.masks is None or len(r.masks) == 0:
            return None
        k = int(r.boxes.conf.argmax())
        m = (r.masks.data[k].cpu().numpy() > 0.5).astype(np.uint8)
        if m.shape != bgr.shape[:2]:
            m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        return m, float(r.boxes.conf[k])

    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
        r = self.mask(bgr)
        if r is None:
            return None
        m, conf = r
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return None
        quad = cv2.boxPoints(cv2.minAreaRect(max(cs, key=cv2.contourArea))).astype(np.float64)
        h, w = bgr.shape[:2]
        mg = self.margin
        if (quad < mg).any() or (quad[:, 0] > w - 1 - mg).any() or (quad[:, 1] > h - 1 - mg).any():
            return None  # cut by the frame border: not a reliable outline
        return quad, conf
