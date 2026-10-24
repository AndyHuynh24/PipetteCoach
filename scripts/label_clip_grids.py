"""Check / fix the plate grid of every train/val dataset clip (Plate_<n>_clip_<m> Topview), one grid per clip (the plate
is still within a clip). Same window and keys as scripts/label_well_corners.py; the proposed grid (orange) is the
auto label from scripts/autolabel_clips_tracker.py. Press n to see another frame of the same clip (5 views across
the clip) when a pipette or hand hides a corner - the grid stays.
Labels (saved on every Enter / x): data/annotations/well_corners_transfyr.json {"clips": [...], "labels":
{str(k * 1000): {H, corners, source}}} for clip k. Test plates and excluded plates/clips are not shown. Keep local.

    python scripts/label_clip_grids.py [--auto data/processed/plate_autolabels/tracker_v3.json]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_well_corners import WellCornerTool  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import clips_for_split  # noqa: E402
from pipette_coach.localizer.session import clip_path  # noqa: E402

VIEWS = 5


def clip_reader(clips: list[str]):
    """read(k * 1000 + v): frame v of VIEWS spread over clip k."""

    def read(code: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(clip_path(clips[code // 1000])))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((code % 1000 + 0.5) / VIEWS * n))
        ok, f = cap.read()
        return f if ok else np.zeros((1080, 1920, 3), np.uint8)

    return read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", default="data/processed/plate_autolabels/tracker_v3.json")
    ap.add_argument("--snapshot", default=None, help="render the first unlabelled clip to this image and exit")
    a = ap.parse_args()
    clips = clips_for_split("train") + clips_for_split("val")
    auto = json.loads((REPO_ROOT / a.auto).read_text())
    props = {
        k * 1000: (np.array(auto[c]["H"]) if auto.get(c, {}).get("H") else None, None) for k, c in enumerate(clips)
    }
    read = clip_reader(clips)
    f0 = read(0)
    out = REPO_ROOT / "data" / "annotations" / "well_corners_transfyr.json"
    tool = WellCornerTool([k * 1000 for k in range(len(clips))], read, 1000.0, out, load_config("localizer"),
                          (f0.shape[1], f0.shape[0]), props)  # fmt: skip
    if tool.lab.get("clips", clips) != clips:
        sys.exit(f"{out} was made for another clip list; move it away first")
    tool.lab["clips"] = clips
    tool.n_alt = VIEWS
    if a.snapshot:
        cv2.imwrite(a.snapshot, tool.render())
        return
    cv2.namedWindow("well corners", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("well corners", tool.on_mouse)
    while True:
        cv2.imshow("well corners", tool.render())
        k = cv2.waitKey(30)
        if k != -1 and not tool.key(k & 0xFF):
            break
    cv2.destroyAllWindows()
    src = [v["source"] for v in tool.lab["labels"].values()]
    print(f"saved -> {out}  {len(src)} clips: " + ", ".join(f"{s} {src.count(s)}" for s in sorted(set(src))))


if __name__ == "__main__":
    main()
