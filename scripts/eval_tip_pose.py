"""Task metrics of a tip-end model on one split of a pose_<data> dataset (scripts/build_tip_dataset.py).

Per frame the most confident tip is taken (single-channel pipettes: one tip). Unit = one well spacing: from the
plate grid on pipette1 (data/processed/tip_data/pipette1_grids.npy), elsewhere from the tip box (0.8 spacing).
  tip_found       labelled frames where the top tip is >= conf AND within 0.5 spacing of a labelled tip end
  error           distance top tip -> nearest labelled tip end, in spacings (median / 90th percentile)
  false_alarm     "no tip" frames with any tip >= conf
  correct_well    pipette1, frames whose labelled tip end is on a well: the top tip is >= conf and maps to the
                  SAME well through the frame's grid (the number that matters for the app)
The confidence threshold is chosen on val (--conf auto: best tip_found - false_alarm); pass it for test.

    python scripts/eval_tip_pose.py --model models/tip_all/weights/best.pt --data tips_all --split val
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.localizer.geometry import pitch_px  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tips import well_of  # noqa: E402

HIT = 0.5  # spacings


def predict_all(model, files, imgsz, device):
    preds = {}
    for k in range(0, len(files), 16):
        rs = model.predict([str(f) for f in files[k : k + 16]], imgsz=imgsz, conf=0.05, verbose=False, device=device)
        for f, r in zip(files[k : k + 16], rs, strict=False):
            kp = r.keypoints.xy.cpu().numpy()[:, 0] if r.keypoints is not None and len(r.boxes) else np.zeros((0, 2))
            preds[f.stem] = (r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0), kp)
    return preds


def score(rows, thr):
    pos = [r for r in rows if r["gt"]]
    neg = [r for r in rows if not r["gt"]]
    found = [r["top_conf"] >= thr and r["err"] <= HIT for r in pos]
    fa = [r["max_conf"] >= thr for r in neg]
    err = [r["err"] for r in pos if r["top_conf"] >= thr]
    wells = [r for r in pos if r.get("gt_well") is not None]
    ok_w = [r["top_conf"] >= thr and r["pred_well"] == r["gt_well"] for r in wells]
    return {
        "tip_frames": len(pos), "tip_found": float(np.mean(found)) if pos else None,
        "error_median": float(np.median(err)) if err else None,
        "error_p90": float(np.percentile(err, 90)) if err else None,
        "no_tip_frames": len(neg), "false_alarm": float(np.mean(fa)) if neg else None,
        "well_frames": len(wells), "correct_well": float(np.mean(ok_w)) if wells else None,
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="tips_all")
    ap.add_argument("--split", default="val", help="train / val / test, or all")
    ap.add_argument("--videos", default="", help="only these videos (comma-separated), e.g. pipette3")
    ap.add_argument("--conf", default="auto")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO

    tcfg = load_config("tip_data")
    root = PROCESSED_DIR / f"pose_{a.data}"
    crops = json.loads((root / "crops.json").read_text())
    splits = ["train", "val", "test"] if a.split == "all" else [a.split]
    files = sorted(f for sp in splits for f in (root / "images" / sp).glob("*.jpg"))
    if a.videos:
        files = [f for f in files if f.stem.rsplit("_f", 1)[0] in a.videos.split(",")]
    preds = predict_all(YOLO(a.model), files, a.imgsz, a.device)
    grids = {}
    rows = []
    for f in files:
        vname, fno = f.stem.rsplit("_f", 1)
        fno = int(fno)
        x0, y0, x1, y1 = crops[f.stem]
        lab = (root / "labels" / f.parent.name / f"{f.stem}.txt").read_text()
        gt = [tuple(map(float, ln.split()[5:7])) for ln in lab.splitlines() if ln.strip()]
        gt = [(u * (x1 - x0) + x0, v * (y1 - y0) + y0) for u, v in gt]
        conf, kp = preds[f.stem]
        kp = kp + [x0, y0]
        if vname not in grids:
            g = PROCESSED_DIR / "tip_data" / f"{vname}_grids.npy"
            grids[vname] = np.load(g) if g.exists() else None
        H = grids[vname][fno] if grids[vname] is not None else None
        H = None if H is None or np.isnan(H).any() else H
        unit = pitch_px(H) if H is not None else tcfg["tip_box_frac"] * max(x1 - x0, y1 - y0) / 0.8
        r = {
            "file": f.name,
            "split": f.parent.name,
            "video": vname,
            "gt": bool(gt),
            "max_conf": float(conf.max()) if len(conf) else 0.0,
        }
        if gt:
            k = int(np.argmax(conf)) if len(conf) else None
            r["top_conf"] = float(conf[k]) if k is not None else 0.0
            r["err"] = float(min(np.hypot(*(kp[k] - g)) for g in gt) / unit) if k is not None else 1e9
            if H is not None:
                r["gt_well"] = well_of(H, gt[0])
                r["pred_well"] = well_of(H, kp[k]) if k is not None else None
        rows.append(r)
    if a.conf == "auto":
        cands = np.round(np.arange(0.1, 0.91, 0.05), 2)
        sc = [(score(rows, t), t) for t in cands]
        thr = max(sc, key=lambda s: (s[0]["tip_found"] or 0) - (s[0]["false_alarm"] or 0))[1]
    else:
        thr = float(a.conf)
    m = {"model": a.model, "data": a.data, "split": a.split, "conf": float(thr), "all": score(rows, thr)}
    for vname in sorted({r["video"] for r in rows}):
        m[vname] = score([r for r in rows if r["video"] == vname], thr)
    run = new_run_dir(a.name or f"tip-eval-{a.split}")
    write_metrics(run, m)
    (run / "rows.json").write_text(json.dumps(rows))
    # worst cases by eye: missed / far-off tips and false alarms
    bad = sorted([r for r in rows if r["gt"] and (r["top_conf"] < thr or r["err"] > HIT)], key=lambda r: -r["err"])
    bad += [r for r in rows if not r["gt"] and r["max_conf"] >= thr]
    tiles = []
    for r in bad[:30]:
        img = cv2.imread(str(root / "images" / r["split"] / r["file"]))
        conf, kp = preds[Path(r["file"]).stem]
        for c, p in zip(conf, kp, strict=False):
            if c >= thr:
                cv2.drawMarker(img, tuple(int(v) for v in p), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 30, 3)
        lab = (root / "labels" / r["split"] / r["file"].replace(".jpg", ".txt")).read_text().split()
        if lab:
            h, w = img.shape[:2]
            cv2.drawMarker(img, (int(float(lab[5]) * w), int(float(lab[6]) * h)), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        t = cv2.resize(img, (300, 300))
        cv2.putText(t, r["file"][:22], (4, 20), 0, 0.5, (0, 255, 255), 1)
        tiles.append(t)
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imwrite(str(run / "errors.jpg"), np.vstack([np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]))
    (run / "notes.md").write_text(
        f"# Tip model on {a.split}\n\nred + = labelled tip end, green x = model (errors.jpg).\n\n```\n"
        f"{json.dumps(m, indent=1)}\n```\n"
    )
    print(json.dumps(m, indent=1), "\n->", run)


if __name__ == "__main__":
    main()
