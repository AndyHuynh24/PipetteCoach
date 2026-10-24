"""Tip-end keypoint data: CVAT points, time splits, crops and YOLO pose label lines (configs/tip_data.yaml)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from pipette_coach.config import REPO_ROOT, load_config


def video_path(name: str) -> Path:
    """Own video file: the `video:` of configs/tip_data.yaml (e.g. pipette5.mov), else data/raw/<name>.mp4."""
    v = load_config("tip_data")["videos"].get(name, {})
    return REPO_ROOT / v["video"] if "video" in v else REPO_ROOT / "data" / "raw" / f"{name}.mp4"


# Liquid colours for recolouring the liquid in the tip: (OpenCV hue 0-180, saturation gain)
LIQUID_COLOURS = {"red": (176, 1.0), "orange": (10, 1.0), "yellow": (27, 1.0), "green": (60, 1.0),
                  "lightblue": (95, 1.0), "blue": (115, 1.0), "purple": (140, 1.0), "clear": (0, 0.15)}  # fmt: skip
LIQUID_ROI = 140  # px around the tip end where liquid is searched (1280x720: longer than the visible liquid column)
LIQUID_WIN = 14  # a blob counts as tip liquid if it comes this close to the labelled tip end


def tip_liquid_mask(img: np.ndarray, tip: tuple[float, float], liquid: list[int]) -> np.ndarray:
    """Float mask 0..1 of the liquid in the tip whose end is at `tip`. liquid = [hue_lo, hue_hi, sat_min] (OpenCV
    hue, lo > hi wraps through red; configs/tip_data.yaml). Only coloured blobs touching the tip end are kept, so
    hands, wells and cups are left alone."""
    lo, hi, smin = liquid
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    hue = ((h >= lo) | (h <= hi)) if lo > hi else ((h >= lo) & (h <= hi))
    x, y = int(tip[0]), int(tip[1])
    disc = np.zeros(h.shape, np.uint8)
    cv2.circle(disc, (x, y), LIQUID_ROI, 1, -1)
    m = (hue & (s >= smin) & (v >= 60) & (disc > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    w = LIQUID_WIN
    near = set(np.unique(lab[max(y - w, 0) : y + w + 1, max(x - w, 0) : x + w + 1])) - {0}
    keep = [i for i in near if stats[i, cv2.CC_STAT_AREA] >= 20]
    return cv2.GaussianBlur(np.isin(lab, keep).astype(np.float32), (0, 0), 1.0)


def liquid_centre(liquid: list[int]) -> float:
    lo, hi, _ = liquid
    return ((lo + (hi + 180 if lo > hi else hi)) / 2) % 180


def recolour(img: np.ndarray, mask: np.ndarray, centre: float, hue: int, sat_gain: float) -> np.ndarray:
    """Move the masked pixels' hue from around `centre` to around `hue` (each keeps its offset, so the meniscus and
    reflections stay); sat_gain < 1 fades the colour (clear liquid)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    d = (hsv[..., 0] - centre + 90) % 180 - 90
    hsv[..., 0] = (hue + d) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_gain, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    m = mask[..., None]
    return (img * (1 - m) + out * m).astype(np.uint8)


def parse_cvat_points(xml: Path) -> tuple[dict[int, list[tuple[float, float]]], int]:
    """CVAT for images 1.1 export of a video task -> ({frame id: [(x, y) tip ends]}, number of frames). Label name
    "tip" in any case; points marked outside are ignored. Frames without points are not in the dict."""
    root = ET.parse(xml).getroot()
    tips: dict[int, list[tuple[float, float]]] = {}
    images = root.findall("image")
    for im in images:
        pts = [tuple(map(float, xy.split(",")))
               for p in im.findall("points") if p.get("label", "").lower() == "tip" and p.get("outside", "0") == "0"
               for xy in p.get("points").split(";")]  # fmt: skip
        pts = list(dict.fromkeys(pts))  # a double click stores the exact same point twice (pipette5)
        if pts:
            tips[int(im.get("id"))] = pts
    return tips, len(images)


def mask_box(el: ET.Element, open_px: int = 15) -> tuple[float, float, float, float]:
    """Box (x0, y0, x1, y1) of a CVAT <mask> (RLE inside its left/top/width/height box, 0-runs first): only the
    largest part, opened, so stray blobs of the semi-automatic mask (SAM) do not grow the box."""
    left, top, w, h = (int(float(el.get(k))) for k in ("left", "top", "width", "height"))
    flat = np.zeros(w * h, np.uint8)
    pos, val = 0, 0
    for run in map(int, el.get("rle").split(",")):
        flat[pos : pos + run] = val
        pos, val = pos + run, val ^ 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(flat.reshape(h, w))
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    ys, xs = np.nonzero(opened if opened.any() else m)
    return float(left + xs.min()), float(top + ys.min()), float(left + xs.max() + 1), float(top + ys.max() + 1)


def parse_cvat_plates(xml: Path, drop: tuple = ()) -> dict[int, tuple[float, float, float, float]]:
    """{frame id: plate box (x0, y0, x1, y1)} from "Plate" boxes or masks of a CVAT for images 1.1 export (one plate
    per frame). Frames in `drop` (checked by eye as wrong) are left out."""
    out = {}
    for im in ET.parse(xml).getroot().findall("image"):
        i = int(im.get("id"))
        for el in im:
            if el.get("label", "").lower() != "plate" or el.get("outside", "0") != "0" or i in drop:
                continue
            if el.tag == "box":
                out[i] = tuple(float(el.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            elif el.tag == "mask":
                out[i] = mask_box(el)
    return out


def split_of(t: float, vcfg: dict) -> str:
    return "train" if t < vcfg["val_from"] else "val" if t < vcfg["test_from"] else "test"


def in_skip(t: float, vcfg: dict) -> bool:
    return any(a <= t < b for a, b in vcfg.get("skip", []))


def negative_candidates(labelled, n_frames: int, fps: float, gap_s: float, vcfg: dict, n: int) -> list[int]:
    """n frames spread evenly over the video, each at least gap_s from every labelled frame and outside skip
    ranges. Candidates only: each one is checked by eye before it is used as "no tip"."""
    lab = np.array(sorted(labelled))
    gap = gap_s * fps
    ok = [f for f in range(n_frames)
          if (len(lab) == 0 or np.abs(lab - f).min() > gap) and not in_skip(f / fps, vcfg)]  # fmt: skip
    if len(ok) <= n:
        return ok
    return [ok[int(i)] for i in np.linspace(0, len(ok) - 1, n)]


def plate_crop_box(outline: np.ndarray, margin: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Plate outline (4, 2) + margin x its size on each side, clipped to the frame -> (x0, y0, x1, y1)."""
    lo, hi = outline.min(0), outline.max(0)
    m = (hi - lo) * margin
    x0, y0 = np.maximum(lo - m, 0).astype(int)
    x1, y1 = np.minimum(hi + m, [w, h]).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


def random_crop_box(rng, frac: tuple[float, float], w: int, h: int, keep=None) -> tuple[int, int, int, int]:
    """Square crop, side = U(frac) x the frame's short side. keep = (x, y): the point lies in the inner 70 %."""
    s = int(rng.uniform(*frac) * min(w, h))
    if keep is None:
        x0, y0 = rng.integers(0, w - s + 1), rng.integers(0, h - s + 1)
    else:
        x0 = int(np.clip(keep[0] - rng.uniform(0.15, 0.85) * s, 0, w - s))
        y0 = int(np.clip(keep[1] - rng.uniform(0.15, 0.85) * s, 0, h - s))
    return int(x0), int(y0), int(x0 + s), int(y0 + s)


def pose_lines(tips, box: tuple[int, int, int, int], box_frac: float) -> list[str] | None:
    """YOLO pose lines (class 0, box, 1 visible keypoint) for tip ends inside the crop box; None if a tip end
    falls outside the crop (frame not usable as cropped)."""
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    side = box_frac * max(cw, ch)
    out = []
    for x, y in tips:
        u, v = x - x0, y - y0
        if not (0 <= u < cw and 0 <= v < ch):
            return None
        out.append(f"0 {u / cw:.6f} {v / ch:.6f} {side / cw:.6f} {side / ch:.6f} {u / cw:.6f} {v / ch:.6f} 2")
    return out


def well_of(H: np.ndarray, pt, max_dist: float = 0.5) -> tuple[int, int] | None:
    """Image point -> (column 0-11, row 0-7) of the nearest well on the grid H (canonical well coords -> image),
    or None if it is more than max_dist well spacings from every well centre (off the plate / between wells)."""
    p = np.linalg.inv(H) @ np.array([pt[0], pt[1], 1.0])
    u, v = p[:2] / p[2]
    c, r = int(np.clip(round(u), 0, 11)), int(np.clip(round(v), 0, 7))
    return (c, r) if np.hypot(u - c, v - r) <= max_dist else None


class DwellDetector:
    """Dispense events from per-frame tip wells: the tip stays on the SAME well for >= min_s -> one event for that
    well (fired once per visit). Gaps up to gap_s (missed detections, skipped frames) do not end a visit. Coming
    back to the well that just fired within refractory_s of last seeing it continues that visit (a tip lifted out of
    view for a moment); a real second dispense needs a trip to the liquid first, which takes seconds.

    One dispense per trip (trip_gap_s set): while the tip stays on the plate (no gap > trip_gap_s), a stop in another
    well that lasts LONGER than the reported one replaces it (the tip paused next to the target on its way in, e.g.
    0.4 s on E10, then 1.5 s on F10 = F10). update() then returns the new well with self.revised = True: the caller
    replaces its last event instead of adding one. trip_gap_s=1.0: on pipette1 every value 0.3-1.75 s gives 32/32
    dispenses right and no false ones (without it: 2 false); 2 s merges two quick trips (scripts/eval_dispenses.py).
    trip_gap_s=None: every qualifying stop is its own dispense."""

    def __init__(self, min_s: float = 0.4, gap_s: float = 0.25, refractory_s: float = 1.5,
                 trip_gap_s: float | None = 1.0):  # fmt: skip
        self.min_s, self.gap_s, self.refractory_s, self.trip_gap_s = min_s, gap_s, refractory_s, trip_gap_s
        self.well, self.start, self.last, self.fired = None, 0.0, -1e9, False
        self.recent: dict = {}  # well that fired -> last time the tip was seen on it
        self.trip: list | None = None  # [well, stop length] reported in the current trip
        self.revised = False

    def progress(self, t: float) -> float:
        """Share of min_s the current visit has lasted (0 when no tip on a well)."""
        if self.well is None or t - self.last > self.gap_s:
            return 0.0
        return min((self.last - self.start) / self.min_s, 1.0)

    def update(self, t: float, well) -> tuple | None:
        self.revised = False
        if self.trip_gap_s is not None and t - self.last > self.trip_gap_s:
            self.trip = None  # the tip left the plate: next stop starts a new trip
        if well is not None:
            if well != self.well or t - self.last > self.gap_s:
                back = t - self.recent.get(well, -1e9) <= self.refractory_s
                self.well, self.start, self.fired = well, t, back
            self.last = t
            if well in self.recent:
                self.recent[well] = t
            dur = self.last - self.start
            if self.trip is not None and self.trip[0] == well:
                self.trip[1] = max(self.trip[1], dur)
                self.fired = True  # back on the reported well: same dispense
            if not self.fired and dur >= self.min_s - 1e-6:
                if self.trip_gap_s is None or self.trip is None:
                    self.fired, self.recent[well] = True, t
                    self.trip = [well, dur] if self.trip_gap_s is not None else None
                    return well
                if dur > self.trip[1]:  # longer stop than the reported one in this trip: it was the dispense
                    self.fired, self.recent[well], self.revised = True, t, True
                    self.trip = [well, dur]
                    return well
        elif t - self.last > self.gap_s:
            self.well = None
        return None
