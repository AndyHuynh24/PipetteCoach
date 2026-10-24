"""Convert a CVAT export ("CVAT for images 1.1", annotations.xml) of tip-end points into a YOLO POSE dataset.

Two ways to annotate (docs/tip_annotation_guide.md):
  --video  a CVAT task made from the WHOLE VIDEO: only frames with a "tip" point (or a "no_tip" tag) are used;
           every other frame is ignored (NOT taken as "no tip": a skipped frame may contain a tip). The frames
           are extracted from the video here (frame number = --frame-start + CVAT frame id * --frame-step).
  --images a CVAT task made from extracted images: every frame is used; no points = no tip.
Label "tip" = Points, one point per visible tip END (multichannel: one per tip); tag "unclear" = left out; tag
"no_tip" = explicit "no tip in this frame" example. Each tip becomes one object of class "tip": a small box centred
on it (0.8 well spacing, measured from the plate detector box) + one keypoint (kpt_shape [1, 3]).
Split by TIME: t < --train-until -> train, t >= --test-from -> test (never used for training or model choice), in
between -> val. A contact sheet (check_labels.jpg) is written to check the labels by eye.

    python scripts/cvat_to_yolo_pose.py --xml data/own/tips_cvat/annotations.xml --video data/raw/pipette1.mp4
"""

import argparse
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_plate_det import detect  # noqa: E402

from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402

TIP_BOX_PITCH = 0.8


def parse(xml: Path) -> list[dict]:
    """[{"id": CVAT frame id, "name", "w", "h", "tips": [(x, y)], "unclear": bool, "no_tip": bool}]"""
    out = []
    for im in ET.parse(xml).getroot().iter("image"):
        tips = []
        for p in im.iter("points"):
            if p.get("label") == "tip" and p.get("outside", "0") == "0":
                tips += [tuple(map(float, xy.split(","))) for xy in p.get("points").split(";")]
        tags = {t.get("label") for t in im.iter("tag")}
        out.append({"id": int(im.get("id")), "name": Path(im.get("name")).name, "w": int(im.get("width")),
                    "h": int(im.get("height")), "tips": tips, "unclear": "unclear" in tags,
                    "no_tip": "no_tip" in tags})  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--images", default=None, help="image task: folder with the annotated images")
    ap.add_argument("--video", default=None, help="video task: the video that was uploaded to CVAT")
    ap.add_argument("--frame-step", type=int, default=1, help="video task: 'Frame step' set when creating it")
    ap.add_argument("--frame-start", type=int, default=0, help="video task: 'Start frame' set when creating it")
    ap.add_argument("--name", default="tips")
    ap.add_argument("--train-until", type=float, default=300.0)
    ap.add_argument("--test-from", type=float, default=390.0)
    a = ap.parse_args()
    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det = PlateDetector(cfg)
    long_p = cfg["plate_edges"]["plate_size_pitch"][0]
    if not (a.images or a.video):
        sys.exit("give --video (task made from the video) or --images (task made from extracted images)")
    ann = parse(Path(a.xml))
    out = PROCESSED_DIR / f"pose_{a.name}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    stats = {"train": [0, 0], "val": [0, 0], "test": [0, 0], "unclear": 0, "missing_image": 0}
    sheet = []
    cap = cv2.VideoCapture(a.video) if a.video else None
    fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
    stem = Path(a.video).stem if a.video else None
    stats["skipped_unannotated"] = 0
    for e in sorted(ann, key=lambda x: x["id"]):
        if e["unclear"]:
            stats["unclear"] += 1
            continue
        if cap is not None:
            if not e["tips"] and not e["no_tip"]:
                stats["skipped_unannotated"] += 1  # not looked at / skipped: never used as "no tip"
                continue
            fno = a.frame_start + e["id"] * a.frame_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, img = cap.read()
            if not ok:
                stats["missing_image"] += 1
                continue
            t = fno / fps
            name = f"{stem}_t{t:06.1f}s_f{fno:05d}.jpg"
        else:
            name = e["name"]
            src = Path(a.images) / name
            if not src.exists():
                stats["missing_image"] += 1
                continue
            img = cv2.imread(str(src))
            t = float(re.search(r"_t(\d+\.\d)s", name).group(1))
        split = "train" if t < a.train_until else "test" if t >= a.test_from else "val"
        h, w = img.shape[:2]
        b = detect(det, img)
        pitch = (max(b[0][2] - b[0][0], b[0][3] - b[0][1]) / long_p) if len(b) else 0.017 * w
        half = TIP_BOX_PITCH * pitch / 2
        lines = []
        for x, y in e["tips"]:
            lines.append(f"0 {x / w:.6f} {y / h:.6f} {2 * half / w:.6f} {2 * half / h:.6f} {x / w:.6f} {y / h:.6f} 2")
        for d in ("images", "labels"):
            (out / d / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text("".join(f"{ln}\n" for ln in lines))
        stats[split][0 if lines else 1] += 1
        if lines and len(sheet) < 24:
            vis = img.copy()
            for x, y in e["tips"]:
                cv2.circle(vis, (int(x), int(y)), int(half), (0, 0, 255), 2)
                cv2.drawMarker(vis, (int(x), int(y)), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cx, cy = np.mean(e["tips"], axis=0).astype(int)
            crop = vis[max(cy - 200, 0) : cy + 200, max(cx - 250, 0) : cx + 250]
            sheet.append(cv2.resize(crop, (300, 240)))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nkpt_shape: [1, 3]\n"
        "flip_idx: [0]\nnames:\n  0: tip\n"
    )
    if sheet:
        while len(sheet) % 6:
            sheet.append(np.zeros_like(sheet[0]))
        cv2.imwrite(
            str(out / "check_labels.jpg"), np.vstack([np.hstack(sheet[i : i + 6]) for i in range(0, len(sheet), 6)])
        )
    print({k: (f"{v[0]} with tips / {v[1]} without" if isinstance(v, list) else v) for k, v in stats.items()},
          "->", out)  # fmt: skip


if __name__ == "__main__":
    main()
