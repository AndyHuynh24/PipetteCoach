"""Hand-label a clip's plate grid by clicking the 4 corner WELLS (for clips marked "neither" in the review).

Click, in order: A1, A12, H12, H1 (well centres; use the zoomed window). Keys: u = undo last click, s = save,
n = next frame (if the corner wells are hidden), q = quit without saving. The grid is fit from the 4 clicks, then
snapped to the wells' opening edges, and saved to configs/grid_labels_manual.json (human labels; the plate is
static within a clip, so one frame labels the whole clip).

    python scripts/label_corners.py Plate_2_clip_0005 [--frame 40]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.localizer.geometry import GRID, project  # noqa: E402
from pipette_coach.localizer.openings import refine_to_openings  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402

CORNERS = np.array([[0, 0], [11, 0], [11, 7], [0, 7]], np.float32)  # A1, A12, H12, H1 in canonical coords
NAMES = ["A1", "A12", "H12", "H1"]
ZOOM = 3
OUT = REPO_ROOT / "configs" / "grid_labels_manual.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frame", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config("localizer")
    frames = read_frames(a.clip)
    fi = a.frame if a.frame is not None else len(frames) // 2
    clicks: list[tuple[float, float]] = []
    view = {"x0": 0, "y0": 0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x / ZOOM + view["x0"], y / ZOOM + view["y0"]))

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("label", on_mouse)
    # crop to the lower-middle of the bench where plates sit; pan with the full frame if needed
    h, w = frames[0].shape[:2]
    view["x0"], view["y0"] = w // 4, h // 3
    while True:
        f = frames[fi]
        crop = f[view["y0"] : view["y0"] + h // 2, view["x0"] : view["x0"] + w // 2]
        img = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
        for k, (x, y) in enumerate(clicks):
            p = (int((x - view["x0"]) * ZOOM), int((y - view["y0"]) * ZOOM))
            cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(img, NAMES[k], (p[0] + 6, p[1] - 6), 0, 0.6, (0, 0, 255), 2)
        if len(clicks) == 4:
            H, _ = cv2.findHomography(CORNERS, np.array(clicks, np.float32), 0)
            for x, y in project(H, GRID):
                cv2.circle(img, (int((x - view["x0"]) * ZOOM), int((y - view["y0"]) * ZOOM)), 3, (0, 255, 0), -1)
        nxt = NAMES[len(clicks)] if len(clicks) < 4 else "s=save"
        msg = f"{a.clip} f{fi}  click {nxt}  (u undo, n next frame, arrows/wasd pan, q quit)"
        cv2.putText(img, msg, (8, 24), 0, 0.6, (0, 255, 255), 2)
        cv2.imshow("label", img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("u") and clicks:
            clicks.pop()
        if k == ord("n"):
            fi = (fi + 5) % len(frames)
        step = 60
        if k == ord("a"):
            view["x0"] = max(0, view["x0"] - step)
        if k == ord("d"):
            view["x0"] = min(w - w // 2, view["x0"] + step)
        if k == ord("w"):
            view["y0"] = max(0, view["y0"] - step)
        if k == ord("x"):
            view["y0"] = min(h - h // 2, view["y0"] + step)
        if k == ord("s") and len(clicks) == 4:
            H, _ = cv2.findHomography(CORNERS, np.array(clicks, np.float32), 0)
            H, n = refine_to_openings(frames[fi], H, cfg)  # snap the hand grid to the well openings
            labels = (
                json.loads(OUT.read_text())
                if OUT.exists()
                else {"_why": "Hand-labelled plate grids (4 corner-well clicks + opening-edge snap)."}
            )
            labels[a.clip] = {"H": H.tolist(), "frame": fi, "opening_refined_wells": n}
            OUT.write_text(json.dumps(labels, indent=1))
            print(f"saved {a.clip} -> {OUT} (opening refit used {n} wells)")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
