"""Candidate dispenses of a tip-labelled video, to build a dispense reference BY EYE (configs/<x>_dispenses.json).

Candidates = stops of the labelled tip end on one well (the user's CVAT labels through the cached live grid,
data/processed/tip_data/<video>_grids.npy from build_tip_dataset.py outlines) lasting >= --min-s. Per candidate a
review tile: the frame mid-stop (tip marked, the well ringed) and the plate around the well before (t0 - 1 s) and
after (t1 + --after-s) the stop, warped flat (3 x 3 wells, the candidate in the middle). Liquid going in shows as
the middle well changing. Nothing here decides: the reference is written by hand from the sheets.
Output: data/processed/dispense_ref/<video>/{sheet_*.jpg, candidates.json}. Keep local (dataset frames).

    python scripts/review_dispense_ref.py --video pipette4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_well_change import stops  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.localizer.geometry import GRID, pitch_px, project  # noqa: E402
from pipette_coach.tips import parse_cvat_points  # noqa: E402
from pipette_coach.well_change import well_cells  # noqa: E402

CELL = 64  # px per well in the warped review crops


def well_name(w) -> str:
    return f"{'ABCDEFGH'[w[1]]}{w[0] + 1}"


def read(cap, f: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(f, 0))
    ok, img = cap.read()
    return img if ok else None


def neighbourhood(img, H, w) -> np.ndarray:
    """3 x 3 wells around w (column, row), warped flat; outside the plate = black."""
    cells = well_cells(img, H, CELL)
    pad = cv2.copyMakeBorder(cells, CELL, CELL, CELL, CELL, cv2.BORDER_CONSTANT)
    c, r = w
    out = pad[r * CELL : (r + 3) * CELL, c * CELL : (c + 3) * CELL].copy()
    cv2.circle(out, (int(1.5 * CELL), int(1.5 * CELL)), int(0.48 * CELL), (0, 255, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="pipette4")
    ap.add_argument("--min-s", type=float, default=0.3)
    ap.add_argument("--after-s", type=float, default=1.5)
    a = ap.parse_args()
    v = load_config("tip_data")["videos"][a.video]
    tips, _ = parse_cvat_points(REPO_ROOT / v["xml"])
    grids = np.load(PROCESSED_DIR / "tip_data" / f"{a.video}_grids.npy")
    cap = cv2.VideoCapture(str(REPO_ROOT / "data" / "raw" / f"{a.video}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = len(grids)
    t = np.arange(n) / fps
    arr = np.full((n, 1, 3), np.nan, np.float32)
    for f, pts in tips.items():
        if f < n and pts:
            arr[f, 0] = [*pts[0], 1.0]
    cand = stops(grids, t, arr, conf=0.5, min_s=a.min_s)
    out = PROCESSED_DIR / "dispense_ref" / a.video
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet_*.jpg"):
        old.unlink()
    rows = []
    for i, s in enumerate(cand):
        f0, f1 = round(s["t0"] * fps), round(s["t1"] * fps)
        mids = sorted(range(f0, f1 + 1), key=lambda f: abs(f - (f0 + f1) / 2))
        fm = next(f for f in mids if np.isfinite(grids[f]).all())  # a stop always has frames with a grid
        H = grids[fm]
        mid = read(cap, fm)
        tip = tips.get(fm, [None])[0]
        c = project(H, GRID[[s["w"][1] * 12 + s["w"][0]]])[0]
        rad = pitch_px(H)
        x0, y0 = np.maximum((c - 4 * rad).astype(int), 0)
        view = mid.copy()
        cv2.circle(view, tuple(np.round(c).astype(int)), int(0.5 * rad), (0, 255, 255), 2)
        if tip is not None:
            cv2.drawMarker(view, (int(tip[0]), int(tip[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        view = cv2.resize(view[y0 : y0 + int(8 * rad), x0 : x0 + int(8 * rad)], (3 * CELL, 3 * CELL))
        tiles = [view]
        for fr in (f0 - round(fps), f1 + round(a.after_s * fps)):
            img = read(cap, min(fr, n - 1))
            Hf = grids[min(max(fr, 0), n - 1)]
            ok = img is not None and np.isfinite(Hf).all()
            tiles.append(neighbourhood(img, Hf, s["w"]) if ok else np.zeros((3 * CELL, 3 * CELL, 3), np.uint8))
        tile = np.hstack(tiles)
        label = f"#{i} {well_name(s['w'])} {s['t0']:.1f}-{s['t1']:.1f}s"
        cv2.putText(tile, label, (4, 18), 0, 0.55, (0, 255, 255), 2)
        cv2.putText(tile, "before", (3 * CELL + 4, 3 * CELL - 6), 0, 0.45, (255, 255, 255), 1)
        cv2.putText(tile, "after", (6 * CELL + 4, 3 * CELL - 6), 0, 0.45, (255, 255, 255), 1)
        rows.append(tile)
    for k in range(0, len(rows), 12):
        chunk = rows[k : k + 12]
        chunk += [np.zeros_like(rows[0])] * (-len(chunk) % 2)
        sheet = np.vstack([np.hstack(chunk[j : j + 2]) for j in range(0, len(chunk), 2)])
        cv2.imwrite(str(out / f"sheet_{k // 12:02d}.jpg"), sheet)
    js = [{"i": i, "well": well_name(s["w"]), "t0": round(s["t0"], 2), "t1": round(s["t1"], 2)}
          for i, s in enumerate(cand)]  # fmt: skip
    (out / "candidates.json").write_text(json.dumps(js, indent=1))
    print(len(cand), "candidates ->", out)


if __name__ == "__main__":
    main()
