"""Compare well-detector models on val plate 4: normal top view and TILTED views (camera tilted 15-35 deg about
the plate centre, truth warped the same way). Same metrics as scripts/train_well_net.py (precision, share of
detections between wells, recall of visible wells, centre error). Never trained on plate 4.

    python scripts/eval_well_nets.py --models models/well_net/best.pt,models/well_net_v2/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_generalization import camera_warp  # noqa: E402
from train_well_net import evaluate, evidence_wells  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clips_for_split  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import plate_bbox, project  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/well_net/best.pt,models/well_net_v2/best.pt")
    a = ap.parse_args()
    cfg = load_config("localizer")
    det = PlateDetector(cfg)
    val = json.loads((PROCESSED_DIR / "plate_autolabels" / "val.json").read_text())
    vt = []
    for clip in clips_for_split("val"):
        good = [x for x in val.get(clip, []) if x["H"] and x["frac"] >= 0.9]
        if good:
            H = np.array(max(good, key=lambda x: x["frac"])["H"])
            vt += [(f, H, evidence_wells(f, H, cfg, det)) for f in read_frames(clip, 8)]
    rng = np.random.default_rng(7)
    vt_tilt = []
    for f, H, ev in vt:
        c = project(H, np.array([[5.5, 3.5]]))[0]
        W = camera_warp(c, 1.0, rng.uniform(-20, 20), rng.uniform(15, 35), rng.uniform(0, 360), c, f=1500.0)
        vt_tilt.append((cv2.warpPerspective(f, W, (f.shape[1], f.shape[0]), borderMode=cv2.BORDER_REPLICATE),
                        W @ H, ev))  # fmt: skip
    res, lines = {}, ["# Well detectors: top view vs tilted (val plate 4)", "",
                      "| model | view | precision | between wells | recall (visible) | centre err px |",
                      "|---|---|---|---|---|---|"]  # fmt: skip
    for mp in [m for m in a.models.split(",") if m]:
        net = WellNet(REPO_ROOT / mp)

        def m_net(img, H, net=net):
            return net.detect(img, plate_bbox(H, 1.0))[0]

        for view, data in (("top", vt), ("tilted 15-35 deg", vt_tilt)):
            r = evaluate(m_net, data, cfg, det, False, np.random.default_rng(1))
            res[f"{mp} | {view}"] = r
            lines.append(f"| {mp} | {view} | {r['precision']:.1%} | {r['in_gaps']:.1%} | {r['recall_visible']:.1%} "
                         f"| {r['centre_err_px'] or 0:.2f} |")  # fmt: skip
            print(lines[-1], flush=True)
    run = new_run_dir("well-net-compare")
    write_metrics(run, res)
    (run / "notes.md").write_text("\n".join(lines) + f"\n\n{len(vt)} frames per view.\n")
    print("->", run)


if __name__ == "__main__":
    main()
