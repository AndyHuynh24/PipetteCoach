"""Live PlateTracker grid vs the user's hand-checked grids (data/annotations/well_corners_<video>.json) on own videos.

The tracker runs over every frame from the start (as live_demo) up to the last labelled frame of the chosen splits
(time split as configs/tip_data.yaml); at each labelled frame its grid is compared with the user's: right (within
0.25 well), off (> 0.25; > 0.5 = a wrong well likely), or no grid. Off-by-labelling (which corner is A1) counts as off.

    python scripts/eval_tracker_labels.py --well-net models/well_net_v4/best.pt --name v4 --splits val
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import grid_error  # noqa: E402
from pipette_coach.localizer.live import PlateTracker  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402


def split_of(t: float, sp: dict) -> str:
    return "train" if t < sp.get("val_from", 1e9) else ("val" if t < sp.get("test_from", 1e9) else "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--well-net", default="models/well_net/best.pt")
    ap.add_argument("--name", default="")
    ap.add_argument("--videos", nargs="+", default=["pipette1", "pipette4"])
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val", "test"])
    a = ap.parse_args()
    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / a.well_net)
    tip_cfg = load_config("tip_data")["videos"]
    metrics = {"well_net": a.well_net, "splits": a.splits}
    for v in a.videos:
        sp = tip_cfg.get(v, {})
        lab = json.loads((REPO_ROOT / "data" / "annotations" / f"well_corners_{v}.json").read_text())["labels"]
        want = {int(k): np.array(e["H"]) for k, e in lab.items()
                if e["H"] is not None and split_of(e["t"], sp) in a.splits}  # fmt: skip
        tr = PlateTracker(cfg, det, net, rows_toward=sp.get("rows_toward"), verify_every=30)
        cap = cv2.VideoCapture(str(REPO_ROOT / "data" / "raw" / f"{v}.mp4"))
        errs = []
        for i in range(max(want) + 1):
            ok, f = cap.read()
            if not ok:
                break
            H = tr.update(f)["H"]
            if i in want:
                errs.append(np.inf if H is None else grid_error(H, want[i], ignore_labelling=False))
        e = np.array(errs)
        fin = np.isfinite(e)
        metrics[v] = {
            "frames": len(e),
            "right_0.25": float((e <= 0.25).mean()),
            "off_0.25": int(((e > 0.25) & fin).sum()),
            "off_0.5": int(((e > 0.5) & fin).sum()),
            "no_grid": int((~fin).sum()),
            "median_err": float(np.median(e[fin])) if fin.any() else None,
        }
        print(v, metrics[v], flush=True)
    run = new_run_dir("tracker-vs-labels" + (f"-{a.name}" if a.name else ""))
    write_metrics(run, metrics)
    (run / "notes.md").write_text(
        f"# Live tracker grid vs hand-checked grids ({a.well_net}, splits {a.splits})\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in metrics.items())
        + "\n\nright = within 0.25 well of the user's grid (labelling included).\n"
    )
    print("->", run)


if __name__ == "__main__":
    main()
