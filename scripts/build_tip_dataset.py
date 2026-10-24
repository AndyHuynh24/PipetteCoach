"""Tip-end YOLO pose dataset from the CVAT exports (configs/tip_data.yaml), in three stages:

outlines    run the live plate tracker over each "crop: plate" video once and cache the plate outline per frame
            (data/processed/tip_data/<video>_{outlines,grids}.npy) - the crop the tip model sees live.
candidates  "no tip" candidates: unlabelled frames >= neg_gap_s from any label, spread over each video; contact
            sheets of exactly what the model would see (data/processed/tip_data/neg_review/) to check BY EYE.
            Only frames confirmed tip-free go into configs/tip_negatives.json (unlabelled != no tip).
build       the dataset: positives = every labelled frame (crop: plate -> plate outline + random margin; crop:
            random -> random square around the tip), negatives = the confirmed frames, split by time.
            -> data/processed/pose_<name>/ with data.yaml (kpt_shape [1, 3]) and check_labels.jpg.

  python scripts/build_tip_dataset.py outlines --videos pipette1
  python scripts/build_tip_dataset.py candidates
  python scripts/build_tip_dataset.py build --videos pipette1,pipette2,pipette3 --name tips_all
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import CONFIG_DIR, REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.tips import (  # noqa: E402
    LIQUID_COLOURS,
    in_skip,
    liquid_centre,
    negative_candidates,
    parse_cvat_plates,
    parse_cvat_points,
    plate_crop_box,
    pose_lines,
    random_crop_box,
    recolour,
    split_of,
    tip_liquid_mask,
    video_path,
)

WORK = PROCESSED_DIR / "tip_data"
TILE = 360


def cvat_plate_outlines(v: dict, n: int, fps: float) -> np.ndarray:
    """Per frame the hand-labelled plate box nearest in time (within plate_max_gap_s) as a 4-corner outline - for a
    video whose plate the live tracker does not find yet (pipette5) and never moves while the tips are labelled."""
    boxes = parse_cvat_plates(REPO_ROOT / v["xml"], drop=tuple(v.get("plate_drop", [])))
    ids = np.array(sorted(boxes))
    out = np.full((n, 4, 2), np.nan, np.float32)
    for i in range(n):
        j = ids[np.abs(ids - i).argmin()]
        if abs(j - i) <= v.get("plate_max_gap_s", 10) * fps:
            x0, y0, x1, y1 = boxes[int(j)]
            out[i] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return out


def read(cap, f: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    return img if ok else None


def outline_at(outlines: np.ndarray, f: int, fps: float) -> np.ndarray | None:
    """Plate outline at frame f, or the nearest one within 0.5 s (tracker had no grid on that frame)."""
    for d in range(int(0.5 * fps) + 1):
        for g in (f - d, f + d):
            if 0 <= g < len(outlines) and not np.isnan(outlines[g]).any():
                return outlines[g]
    return None


def stage_outlines(cfg: dict, videos: list[str]) -> None:
    from pipette_coach.localizer.detector import PlateDetector
    from pipette_coach.localizer.live import PlateTracker
    from pipette_coach.localizer.well_net import WellNet

    lcfg = load_config("localizer")
    lcfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det, net = PlateDetector(lcfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    WORK.mkdir(parents=True, exist_ok=True)
    for name in videos:
        v = cfg["videos"][name]
        if v["crop"] != "plate":
            continue
        cap = cv2.VideoCapture(str(video_path(name)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if v.get("outline") == "cvat_plate":
            out = cvat_plate_outlines(v, n, cap.get(cv2.CAP_PROP_FPS))
            np.save(WORK / f"{name}_outlines.npy", out)
            np.save(WORK / f"{name}_grids.npy", np.full((n, 3, 3), np.nan))  # no grid: no correct-well check
            print(name, "frames with a plate outline (hand labels):", float((~np.isnan(out[:, 0, 0])).mean()))
            continue
        out = np.full((n, 4, 2), np.nan, np.float32)
        hs = np.full((n, 3, 3), np.nan)  # the grid (canonical well coords -> image), for the correct-well check
        tr = PlateTracker(lcfg, det, net, rows_toward=v.get("rows_toward"))
        for i in range(n):
            ok, f = cap.read()
            if not ok:
                break
            o = tr.update(f)
            if o.get("outline") is not None:
                out[i], hs[i] = o["outline"], o["H"]
            if i % 1000 == 0:
                print(f"{name} {i}/{n}", flush=True)
        np.save(WORK / f"{name}_outlines.npy", out)
        np.save(WORK / f"{name}_grids.npy", hs)
        print(name, "frames with a plate outline:", float((~np.isnan(out[:, 0, 0])).mean()))


def stage_candidates(cfg: dict, videos: list[str]) -> None:
    rev = WORK / "neg_review"
    rev.mkdir(parents=True, exist_ok=True)
    cj = WORK / "neg_candidates.json"
    allc = json.loads(cj.read_text()) if cj.exists() else {}
    for name in videos:
        v = cfg["videos"][name]
        for old in rev.glob(f"{name}_*.jpg"):
            old.unlink()
        tips, n = parse_cvat_points(REPO_ROOT / v["xml"])
        cap = cv2.VideoCapture(str(video_path(name)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = min(n, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        outlines = np.load(WORK / f"{name}_outlines.npy") if v["crop"] == "plate" else None
        cand = negative_candidates(tips, n, fps, cfg["neg_gap_s"], v, v["n_neg"])
        tiles, kept = [], []
        for f in cand:
            img = read(cap, f)
            if img is None:
                continue
            if outlines is not None:  # what the model sees: the plate crop at the LARGEST margin used
                o = outline_at(outlines, f, fps)
                if o is None:
                    continue
                x0, y0, x1, y1 = plate_crop_box(o, cfg["plate_margin"][1], img.shape[1], img.shape[0])
                img = img[y0:y1, x0:x1]
            s = TILE / max(img.shape[:2])
            t = np.zeros((TILE, TILE, 3), np.uint8)
            r = cv2.resize(img, None, fx=s, fy=s)
            t[: r.shape[0], : r.shape[1]] = r
            cv2.putText(t, str(f), (4, 22), 0, 0.7, (0, 255, 255), 2)
            tiles.append(t)
            kept.append(f)
        allc[name] = kept
        for k in range(0, len(tiles), 20):
            chunk = tiles[k : k + 20]
            chunk += [np.zeros_like(tiles[0])] * (20 - len(chunk))
            sheet = np.vstack([np.hstack(chunk[i : i + 5]) for i in range(0, 20, 5)])
            cv2.imwrite(str(rev / f"{name}_{k // 20:02d}.jpg"), sheet)
        print(name, len(kept), "candidates")
    cj.write_text(json.dumps(allc))


def stage_build(cfg: dict, videos: list[str], name: str) -> None:
    rng = np.random.default_rng(cfg["seed"])
    neg_file = CONFIG_DIR / "tip_negatives.json"
    negs = json.loads(neg_file.read_text())["confirmed"] if neg_file.exists() else {}
    out = PROCESSED_DIR / f"pose_{name}"
    if out.exists():
        shutil.rmtree(out)
    for d in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (out / d / sp).mkdir(parents=True)
    stats: dict = {}
    crops: dict = {}  # image stem -> crop box (x0, y0, x1, y1) in the video frame (to map predictions back)
    sheet_pos, sheet_neg = [], []
    for vname in videos:
        v = cfg["videos"][vname]
        tips, _ = parse_cvat_points(REPO_ROOT / v["xml"])
        cap = cv2.VideoCapture(str(video_path(vname)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        outlines = np.load(WORK / f"{vname}_outlines.npy") if v["crop"] == "plate" else None
        st = stats[vname] = {sp: [0, 0] for sp in ("train", "val", "test")}
        st |= {"no_crop": 0, "skipped": 0, "recoloured": 0}
        step = v.get("label_step", 1)  # 30 fps runs of near-identical frames: every step-th labelled frame is enough
        items = [(f, tips[f]) for f in sorted(tips) if f % step == 0] + [(f, []) for f in negs.get(vname, [])]
        for f, pts in items:
            if not pts and in_skip(f / fps, v):  # skip ranges: never "no tip"; labelled tips there are kept
                st["skipped"] += 1
                continue
            img = read(cap, f)
            if img is None:
                st["no_crop"] += 1
                continue
            h, w = img.shape[:2]
            lines = None
            for _ in range(5):
                if outlines is not None:
                    o = outline_at(outlines, f, fps)
                    if o is None:
                        break
                    box = plate_crop_box(o, rng.uniform(*cfg["plate_margin"]), w, h)
                else:
                    keep = np.mean(pts, axis=0) if pts else None
                    box = random_crop_box(rng, cfg["random_crop"], w, h, keep)
                lines = pose_lines(pts, box, cfg["tip_box_frac"])
                if lines is not None:
                    break
            if lines is None:
                st["no_crop"] += 1
                continue
            sp = split_of(f / fps, v)
            x0, y0, x1, y1 = box
            crop = img[y0:y1, x0:x1]
            stem = f"{vname}_f{f:05d}"
            cv2.imwrite(str(out / "images" / sp / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / sp / f"{stem}.txt").write_text("".join(f"{ln}\n" for ln in lines))
            st[sp][0 if pts else 1] += 1
            crops[stem] = [int(x0), int(y0), int(x1), int(y1)]
            if pts and "liquid" in v:  # recoloured tip liquid: train random colours, val / test every colour
                m = np.maximum.reduce([tip_liquid_mask(img, p, v["liquid"]) for p in pts])
                if m.max() >= 0.5:
                    names = list(LIQUID_COLOURS)
                    pick = rng.choice(names, cfg["recolour_copies"], replace=False) if sp == "train" else names
                    for c in pick:
                        rc = recolour(img, m, liquid_centre(v["liquid"]), *LIQUID_COLOURS[c])[y0:y1, x0:x1]
                        cv2.imwrite(str(out / "images" / sp / f"{stem}_{c}.jpg"), rc, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        (out / "labels" / sp / f"{stem}_{c}.txt").write_text("".join(f"{ln}\n" for ln in lines))
                        crops[f"{stem}_{c}"] = crops[stem]
                        st["recoloured"] += 1
            vis = crop.copy()
            for x, y in pts:
                cv2.drawMarker(vis, (int(x - x0), int(y - y0)), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
            if pts and rng.random() < 0.05 and len(sheet_pos) < 24:
                sheet_pos.append(cv2.resize(vis, (TILE, TILE)))
            elif not pts and rng.random() < 0.1 and len(sheet_neg) < 12:
                sheet_neg.append(cv2.resize(vis, (TILE, TILE)))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nkpt_shape: [1, 3]\n"
        "flip_idx: [0]\nnames:\n  0: tip\n"
    )
    tiles = sheet_pos + sheet_neg
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
        cv2.imwrite(str(out / "check_labels.jpg"), np.vstack([np.hstack(tiles[i : i + 6])
                                                              for i in range(0, len(tiles), 6)]))  # fmt: skip
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "crops.json").write_text(json.dumps(crops))
    print(json.dumps(stats), "->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["outlines", "candidates", "build"])
    ap.add_argument("--videos", default="pipette1,pipette2,pipette3")
    ap.add_argument("--name", default="tips_all")
    a = ap.parse_args()
    cfg = load_config("tip_data")
    if a.stage == "outlines":
        stage_outlines(cfg, a.videos.split(","))
    elif a.stage == "candidates":
        stage_candidates(cfg, a.videos.split(","))
    else:
        stage_build(cfg, a.videos.split(","), a.name)


if __name__ == "__main__":
    main()
