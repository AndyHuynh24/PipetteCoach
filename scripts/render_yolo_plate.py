"""Render ONLY the YOLO plate layer (box detector or segmentation) on a video, searched at several scales, in
three versions from one detection pass: rgb (normal), gray (black and white frame, overlays in colour) and heat
(where the model thinks the plate is: every candidate down to conf 0.05, its mask - or box for the box detector -
weighted by its confidence, max over candidates, as a colour map over the darkened frame).

Per frame the frame is tried at the scale that worked last, then the other SEARCH_SCALES. Drawn: green = accepted
plate (conf >= detector.min_conf; segmentation: mask shaded + 4-sided fit) with its scale and confidence; thin
orange = weaker candidates. Output: runs/<ts>_yolo-plate[-seg]/yolo_plate_{rgb,gray,heat}.mp4 + metrics.
Keep local: third-party video.

    python scripts/render_yolo_plate.py --video data/raw/pipette1.mp4 --seconds 60
    python scripts/render_yolo_plate.py --video ... --seg models/plate_seg/weights/best.pt   # segmentation model
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_all import encode  # noqa: E402

from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector, boxes_of  # noqa: E402
from pipette_coach.localizer.plate_grid import SEARCH_SCALES  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

STYLES = ("rgb", "gray", "heat")
HEAT_CONF = 0.05


def detect_all(det, img: np.ndarray, conf: float):
    """(quads (n, 4, 2), confs (n,), masks (n, H, W) uint8 or None) at the image's own size. Box detector: OBB
    quads. Segmenter: the mask of each instance and its minimum-area rectangle."""
    if not hasattr(det, "mask"):
        r = det.model.predict(img, imgsz=det.imgsz, conf=conf, verbose=False, device=det.device)[0]
        quads, confs = boxes_of(r)  # oriented (OBB model) or axis-aligned (detect model) boxes
        return quads, confs, None
    r = det.model.predict(img, imgsz=det.imgsz, conf=conf, verbose=False, device=det.device, retina_masks=True)[0]
    if r.masks is None or len(r.masks) == 0:
        return np.zeros((0, 4, 2)), np.zeros(0), None
    masks = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
    quads = []
    for m in masks:
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea) if cs else np.zeros((4, 1, 2), np.int32)
        quads.append(cv2.boxPoints(cv2.minAreaRect(c)))
    return np.array(quads, float).reshape(-1, 4, 2), r.boxes.conf.cpu().numpy(), masks


def heat_map(shape, quads, confs, masks) -> np.ndarray:
    """Max over candidates of (confidence x region), region = mask (segmentation) or filled box (box detector)."""
    h, w = shape
    heat = np.zeros((h, w), np.float32)
    for k in range(len(confs)):
        if masks is not None:
            reg = cv2.resize(masks[k], (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            reg = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(reg, np.round(quads[k]).astype(np.int32), 1)
            reg = reg.astype(bool)
        heat[reg] = np.maximum(heat[reg], confs[k])
    return heat


def bases(f: np.ndarray, heat: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    col = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (np.clip(heat, 0, 1) * 0.75)[..., None]
    dark = (gray * 0.45).astype(np.uint8)
    return {"rgb": f.copy(), "gray": gray, "heat": (dark * (1 - a) + col * a).astype(np.uint8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--detector", default=None, help="plate detector weights (default: configs/localizer.yaml)")
    ap.add_argument("--seg", default=None, help="plate SEGMENTATION weights (e.g. models/plate_seg/weights/best.pt)")
    a = ap.parse_args()
    cfg = load_config("localizer")
    if a.detector:
        cfg["detector"]["weights"] = a.detector
    if a.seg:
        from pipette_coach.localizer.segmenter import PlateSegmenter

        det = PlateSegmenter(cfg, weights=a.seg)
    else:
        det = PlateDetector(cfg)
    kind = "YOLO plate segmentation only" if a.seg else "YOLO plate detector only"
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
    run = new_run_dir("yolo-plate-seg" if a.seg else "yolo-plate")
    writers: dict[str, cv2.VideoWriter] = {}
    last = 1.0
    n = found = 0
    scales_used: dict[float, int] = {}
    t0 = time.time()
    while n < int(a.seconds * fps):
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        lw = max(1, round(f.shape[1] / 900))
        hit, weak_best = None, None
        for s0 in [last] + [s for s in SEARCH_SCALES if s != last]:
            interp = cv2.INTER_AREA if s0 < 1 else cv2.INTER_LINEAR
            g = f if s0 == 1.0 else cv2.resize(f, None, fx=s0, fy=s0, interpolation=interp)
            quads, confs, masks = detect_all(det, g, HEAT_CONF)
            if len(confs) and (weak_best is None or confs.max() > weak_best[2].max()):
                weak_best = (s0, quads / s0, confs, masks)
            if len(confs) and confs.max() >= det.min_conf:
                hit = (s0, quads / s0, confs, masks)
                break
        cand = hit or weak_best
        heat = heat_map(f.shape[:2], *cand[1:]) if cand is not None else np.zeros(f.shape[:2], np.float32)
        imgs = bases(f, heat)
        if hit is not None:
            s0, qs, cs, masks = hit
            k = int(np.argmax(cs))
            last = s0
            found += 1
            scales_used[s0] = scales_used.get(s0, 0) + 1
            mk = None if masks is None else cv2.resize(masks[k], (f.shape[1], f.shape[0]),
                                                       interpolation=cv2.INTER_NEAREST)  # fmt: skip
            status = f"PLATE conf {cs[k]:.2f} (found at scale {s0})"
            for style, img in imgs.items():
                if mk is not None and style != "heat":  # plate pixels, translucent green
                    img[mk > 0] = (0.55 * img[mk > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
                for j in range(len(cs)):
                    if j != k and cs[j] >= 0.1:
                        cv2.polylines(img, [qs[j].astype(np.int32)], True, (0, 165, 255), lw, cv2.LINE_AA)
                cv2.polylines(img, [qs[k].astype(np.int32)], True, (0, 255, 0), 3 * lw, cv2.LINE_AA)
                p = qs[k].min(0).astype(int)
                cv2.putText(img, f"plate {cs[k]:.2f} @ scale {s0}", (p[0], max(30, p[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9 * lw, (0, 255, 0), 2 * lw, cv2.LINE_AA)  # fmt: skip
        else:
            status = "no plate (orange = weak candidates below the acceptance threshold)"
            if weak_best is not None:
                _, qs, cs, _ = weak_best
                for img in imgs.values():
                    for qq, cc in zip(qs, cs, strict=True):
                        if cc >= 0.1:
                            cv2.polylines(img, [qq.astype(np.int32)], True, (0, 165, 255), lw, cv2.LINE_AA)
                            cv2.putText(img, f"{cc:.2f}", tuple(qq.min(0).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7 * lw, (0, 165, 255), lw, cv2.LINE_AA)  # fmt: skip
        for style, img in imgs.items():
            cv2.rectangle(img, (0, 0), (img.shape[1], 40 * lw), (0, 0, 0), -1)
            title = f"t={a.start + (n - 1) / fps:5.1f}s  {kind} [{style}]:  {status}"
            cv2.putText(img, title, (10, 28 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * lw, (255, 255, 255), lw, cv2.LINE_AA)
            if style not in writers:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writers[style] = cv2.VideoWriter(
                    str(run / f"raw_{style}.mp4"), fourcc, fps, (img.shape[1], img.shape[0])
                )
            writers[style].write(img)
        if n % 150 == 0:
            print(f"{n} frames, plate in {found} ({time.time() - t0:.0f}s)", flush=True)
    for style, vw in writers.items():
        vw.release()
        encode(run / f"raw_{style}.mp4", run / f"yolo_plate_{style}.mp4")
    used = {str(k): v for k, v in sorted(scales_used.items())}
    m = {"video": a.video, "model": a.seg or cfg["detector"]["weights"], "frames": n, "plate_frames": found,
         "plate_share": found / max(n, 1), "scales_used": used}  # fmt: skip
    write_metrics(run, m)
    (run / "notes.md").write_text(f"# {kind}\n\nThree versions: rgb, gray, heat. {m}. Third-party video: keep local.\n")
    print("->", run, m)


if __name__ == "__main__":
    main()
