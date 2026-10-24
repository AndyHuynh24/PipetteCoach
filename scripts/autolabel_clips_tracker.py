"""Auto-label the plate grid of every train/val dataset clip (Plate_<n>_clip_<m> Topview) with the live PlateTracker and
a chosen well net, and compare with the trusted references.

Per clip: all frames through a fresh PlateTracker (as live_demo; rows_toward from configs/plate_orientation.json).
Label = the tracker's grid on the last frame that has one; "agree_share" = share of grid frames within 0.25 well of
it (low = the grid jumped). Reference: data/processed/clip_grids.json for clips whose stored grid the ring check
marked OK (data/processed/label_ring_check.json). Test plates and excluded plates/clips are never touched.
Output: data/processed/plate_autolabels/tracker_<name>.json {clip: {H, grid_share, agree_share, ref_error}} and
runs/<ts>_clip-autolabel-<name>/{metrics.json, notes.md, sheet_<split>.jpg} (green = auto label, red = reference
where they differ by > 0.25 well). Keep the sheets local.

    python scripts/autolabel_clips_tracker.py --well-net models/well_net_v3/best.pt --name v3
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clips_for_split, plate_of, plate_rows_toward  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import GRID, grid_error, pitch_px, project  # noqa: E402
from pipette_coach.localizer.live import PlateTracker  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402


def draw(f: np.ndarray, H: np.ndarray | None, ref: np.ndarray | None, text: str) -> np.ndarray:
    f = f.copy()
    if ref is not None and (H is None or grid_error(H, ref, ignore_labelling=False) > 0.25):
        for q in project(ref, GRID):
            cv2.circle(f, tuple(int(v) for v in q), 3, (0, 0, 255), -1)
    if H is not None:
        for q in project(H, GRID):
            cv2.circle(f, tuple(int(v) for v in q), 2, (0, 255, 0), -1)
    c = project(H if H is not None else ref, np.array([[5.5, 3.5]]))[0] if (H is not None or ref is not None) else None
    if c is not None:
        p = pitch_px(H if H is not None else ref)
        x0, y0 = np.clip((c - np.array([9, 7]) * p).astype(int), 0, None)
        f = f[y0 : int(c[1] + 7 * p), x0 : int(c[0] + 9 * p)]
    out = cv2.resize(f, (360, 280))
    cv2.putText(out, text, (4, 16), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--well-net", default="models/well_net_v3/best.pt")
    ap.add_argument("--name", default="v3")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    a = ap.parse_args()
    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / a.well_net)
    refs = json.loads((PROCESSED_DIR / "clip_grids.json").read_text())
    ok = {r["clip"] for r in json.loads((PROCESSED_DIR / "label_ring_check.json").read_text()) if r["verdict"] == "OK"}
    run = new_run_dir(f"clip-autolabel-{a.name}")
    labels, metrics = {}, {"well_net": a.well_net}
    for split in a.splits:
        tiles, errs, n_none = [], [], 0
        for clip in clips_for_split(split):
            tr = PlateTracker(cfg, det, net, rows_toward=plate_rows_toward(plate_of(clip)), verify_every=30)
            frames = read_frames(clip)
            Hs = [tr.update(f)["H"] for f in frames]
            got = [k for k, H in enumerate(Hs) if H is not None]
            H = Hs[got[-1]] if got else None
            ref = np.array(refs[clip]["H"]) if clip in ok and refs.get(clip, {}).get("H") is not None else None
            agree = float(np.mean([grid_error(Hs[k], H, ignore_labelling=False) <= 0.25 for k in got])) if got else 0.0
            e = None if (H is None or ref is None) else grid_error(H, ref, ignore_labelling=False)
            labels[clip] = {"H": None if H is None else np.asarray(H).tolist(), "grid_share": len(got) / len(frames),
                            "agree_share": agree, "ref_error": e}  # fmt: skip
            if ref is not None:
                errs.append(np.inf if e is None else e)
            n_none += H is None
            tag = "no grid" if H is None else ("no ref" if ref is None else f"ref err {e:.2f}")
            tiles.append(draw(frames[got[-1] if got else -1], H, ref, f"{clip} {tag} agree {agree:.0%}"))
            print(clip, tag, f"grid {len(got)}/{len(frames)} agree {agree:.0%}", flush=True)
        errs = np.array(errs)
        metrics[split] = {"clips": len(tiles), "no_grid": int(n_none), "with_reference": len(errs),
                          "match_reference_0.25": float((errs <= 0.25).mean()) if len(errs) else None,
                          "off_by_0.5_or_more": int((errs > 0.5).sum())}  # fmt: skip
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(run / f"sheet_{split}.jpg"),
                    np.vstack([np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]))  # fmt: skip
        print(split, metrics[split], flush=True)
    dst = PROCESSED_DIR / "plate_autolabels" / f"tracker_{a.name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(labels, indent=1))
    write_metrics(run, metrics)
    (run / "notes.md").write_text(
        f"# Dataset clips auto-labelled with the live tracker + {a.well_net}\n\n"
        "Label = tracker grid on the last frame with one. Reference = clip_grids.json, ring-check OK clips only.\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in metrics.items())
        + f"\n\nLabels: {dst.relative_to(REPO_ROOT)}. Sheets: green = auto label, red = reference where they differ. "
        "Keep local (dataset frames).\n"
    )
    print("->", run)


if __name__ == "__main__":
    main()
