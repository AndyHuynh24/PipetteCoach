"""Well-net training labels from the hand-checked grids (scripts/label_well_corners.py): per labelled frame, which wells
are visible (positives) and which are ignored (pipette_coach/localizer/well_labels.py), plus the time split.

Split by TIME as the tip data (configs/tip_data.yaml): t < val_from -> train, val_from <= t < test_from -> val,
later -> test (never looked at here).
Output: data/processed/well_labels/<video>.json {fps, frames: [{idx, t, H, split, used: 96 x 0/1 | null, n_refs}]}
(used null = frame not used: no clean reference) + runs/<ts>_well-labels/<video>_sheet.jpg to check by eye
(green dot = used as a well, red X = changed / covered, grey X = no rim seen). Keep the sheets local.
--transfyr: also the dataset clips (scripts/label_clip_grids.py, one hand-checked grid per clip; the plate is still
within a clip, so every 3rd frame gets that grid and the clip's other frames are the clean reference). Split by
plate (configs/splits.json). -> data/processed/well_labels/transfyr.json {frames: [{clip, frame, H, split, used}]}.

    python scripts/build_well_labels.py --videos pipette1 pipette4 --transfyr
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, load_splits, plate_of  # noqa: E402
from pipette_coach.localizer.geometry import GRID, grid_error, pitch_px, project  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.localizer.well_labels import (  # noqa: E402
    THRESHOLD,
    change,
    flat_lab,
    reference_frames,
    rim_evidence,
)
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

MIN_REFS = 5
REF_PER_CLIP, MIN_REF_CLIPS = 6, 3  # dataset clips


def tile(f: np.ndarray, H: np.ndarray, ev: np.ndarray, ch: np.ndarray | None, text: str) -> np.ndarray:
    f = f.copy()
    g, p = project(H, GRID), pitch_px(H)
    for q, a, d in zip(g, ev, ch if ch is not None else np.full(len(GRID), np.inf), strict=True):
        c = tuple(int(x) for x in q)
        if a and d < THRESHOLD:
            cv2.circle(f, c, int(0.12 * p), (0, 255, 0), -1)
        else:
            col = (0, 0, 255) if d >= THRESHOLD else (200, 200, 200)
            cv2.drawMarker(f, c, col, cv2.MARKER_TILTED_CROSS, int(0.4 * p), 2)
    m = project(H, np.array([[5.5, 3.5]]))[0]
    hw = np.array([7.5, 5.5]) * p
    x0, y0 = np.clip((m - hw).astype(int), 0, None)
    out = cv2.resize(f[y0 : int(m[1] + hw[1]), x0 : int(m[0] + hw[0])], (480, 360))
    cv2.putText(out, text, (6, 22), 0, 0.6, (0, 255, 255), 2)
    return out


def transfyr_labels(cfg: dict, run: Path) -> dict:
    """Clean look of each well = median over REF_PER_CLIP frames of every clip of the same plate whose hand-checked
    grid puts the plate in the same place (within 0.3 well): a pipette held still for a whole clip is then a minority.
    Clips with fewer than MIN_REF_CLIPS such clips are not used. Wells listed in configs/well_label_ignore.json
    (checked by eye: covered in every clip) are never positives."""
    lab = json.loads((REPO_ROOT / "data" / "annotations" / "well_corners_transfyr.json").read_text())
    split_of = {p: k for k, ps in load_splits().items() for p in ps}
    ignore = json.loads((REPO_ROOT / "configs" / "well_label_ignore.json").read_text())
    clips = {}
    for key, e in sorted(lab["labels"].items(), key=lambda kv: int(kv[0])):
        clip = lab["clips"][int(key) // 1000]
        if e["H"] is not None and plate_of(clip) in split_of:
            frames = read_frames(clip, 3)
            H = np.array(e["H"])
            pick = np.linspace(0, len(frames) - 1, min(REF_PER_CLIP, len(frames))).astype(int)
            clips[clip] = (H, frames, [flat_lab(frames[k], H) for k in pick])
    out, tiles, skipped = [], [], []
    for clip, (H, frames, _) in clips.items():
        same = [c for c, (Hc, _, _) in clips.items()
                if plate_of(c) == plate_of(clip) and grid_error(Hc, H, ignore_labelling=False) <= 0.3]  # fmt: skip
        if len(same) < MIN_REF_CLIPS:
            skipped.append(clip)
            print(clip, f"NOT USED: plate in this place in only {len(same)} clip(s)", flush=True)
            continue
        refs = [F for c in same for F in clips[c][2]]
        rows = ignore.get(clip, {}).get("rows", "")
        never = np.isin(GRID[:, 1].astype(int), [ord(r) - ord("A") for r in rows])  # by eye: always covered
        mid = len(frames) // 2
        for k, f in enumerate(frames):
            ev = rim_evidence(f, H, cfg)
            ch = change(flat_lab(f, H), refs)
            used = ev & (ch < THRESHOLD) & ~never
            out.append({"clip": clip, "frame": 3 * k, "H": H.tolist(), "split": split_of[plate_of(clip)],
                        "used": used.astype(int).tolist()})  # fmt: skip
            if k == mid:
                tiles.append(tile(f, H, ev, ch, f"{clip} {split_of[plate_of(clip)]} used {used.sum()}"))
        n_used = np.mean([sum(o["used"]) for o in out[-len(frames) :]])
        print(clip, f"{len(frames)} frames, {len(same)} reference clips, wells used per frame {n_used:.0f}", flush=True)
    dst = PROCESSED_DIR / "well_labels" / "transfyr.json"
    dst.write_text(json.dumps({"frames": out}))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(
        str(run / "transfyr_sheet.jpg"), np.vstack([np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)])
    )
    return {"clips_not_used": skipped} | {sp: {"frames": sum(o["split"] == sp for o in out),
                 "wells_per_frame": float(np.mean([sum(o["used"]) for o in out if o["split"] == sp] or [0]))}
            for sp in ("train", "val")}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=["pipette1", "pipette4"])
    ap.add_argument("--transfyr", action="store_true", help="also the hand-checked dataset clips")
    a = ap.parse_args()
    cfg = load_config("localizer")
    splits = load_config("tip_data")["videos"]
    run = new_run_dir("well-labels")
    summary = {}
    for v in a.videos:
        lab = json.loads((REPO_ROOT / "data" / "annotations" / f"well_corners_{v}.json").read_text())["labels"]
        cap = cv2.VideoCapture(str(REPO_ROOT / "data" / "raw" / f"{v}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        items = [(int(k), np.array(e["H"]), float(e["t"])) for k, e in sorted(lab.items(), key=lambda kv: int(kv[0]))
                 if e["H"] is not None]  # fmt: skip
        frames, flats = {}, {}
        for idx, H, _ in items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            frames[idx] = cap.read()[1]
            flats[idx] = flat_lab(frames[idx], H)
        sp = splits.get(v, {})
        out, tiles = [], []
        for k, (idx, H, t) in enumerate(items):
            split = "train" if t < sp.get("val_from", 1e9) else ("val" if t < sp.get("test_from", 1e9) else "test")
            refs = reference_frames(items, k)
            ev = rim_evidence(frames[idx], H, cfg)
            ch = change(flats[idx], [flats[items[j][0]] for j in refs]) if len(refs) >= MIN_REFS else None
            used = None if ch is None else (ev & (ch < THRESHOLD))
            out.append({"idx": idx, "t": t, "H": H.tolist(), "split": split, "n_refs": len(refs),
                        "used": None if used is None else used.astype(int).tolist()})  # fmt: skip
            if split != "test":
                txt = f"{t:.0f}s {split} " + (
                    f"used {used.sum()}" if used is not None else f"NOT USED ({len(refs)} refs)"
                )
                tiles.append(tile(frames[idx], H, ev, ch, txt))
        dst = PROCESSED_DIR / "well_labels" / f"{v}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps({"fps": fps, "frames": out}))
        if tiles:
            while len(tiles) % 6:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(run / f"{v}_sheet.jpg"),
                        np.vstack([np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]))  # fmt: skip
        s = {}
        for split in ("train", "val", "test"):
            fr = [f for f in out if f["split"] == split]
            u = [f for f in fr if f["used"] is not None]
            s[split] = {"labelled": len(fr), "used": len(u),
                        "wells_per_frame": float(np.mean([sum(f["used"]) for f in u])) if u else 0.0}  # fmt: skip
        summary[v] = s
        print(v, s, flush=True)
    if a.transfyr:
        summary["transfyr"] = transfyr_labels(cfg, run)
    write_metrics(run, {"videos": summary, "threshold": THRESHOLD, "min_refs": MIN_REFS})
    (run / "notes.md").write_text(
        "# Well-net training labels from the hand-checked grids\n\n"
        "Rule: pipette_coach/localizer/well_labels.py (rim seen AND unchanged vs the same well in other labelled "
        f"frames at the same place, threshold {THRESHOLD}); frames with < {MIN_REFS} reference frames not used. "
        "Split by time as configs/tip_data.yaml.\n\n"
        + "\n".join(f"- {v}: {s}" for v, s in summary.items())
        + "\n\nSheets: green dot = used as a well, red X = changed (covered), grey X = no rim seen. Test frames "
        "are not drawn. Keep local.\n"
    )
    print("->", run)


if __name__ == "__main__":
    main()
