"""Video of the tip models on clips: what B sees per frame and how B and C decide the well.

Per frame: [plate view + model-B heatmap, peak = red cross] [12 x 8 well map: B's summed score per well so far
(the clip decision builds up), true dispensed wells ringed green] [C's per-well score this frame].
Output: runs/<ts>_tip-heatmaps/tip_heatmaps.mp4  (dataset frames: keep local). Test plates refused.

    python scripts/viz_tip_heatmaps.py --clips Plate_4_clip_0003,Plate_4_clip_0009 [--b models/tip_b/best.pt]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_all import encode  # noqa: E402
from train_tip_c import cell_index, well_scores  # noqa: E402

from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import load_splits, plate_of  # noqa: E402
from pipette_coach.localizer.detector import _default_device  # noqa: E402
from pipette_coach.localizer.well_net import build_model, peaks, to_tensor  # noqa: E402
from pipette_coach.protocol.wells import index_well, well_str  # noqa: E402
from pipette_coach.runs import new_run_dir  # noqa: E402
from pipette_coach.tracking import plate_view as pv  # noqa: E402
from pipette_coach.tracking.tip_data import clip_wells, load_clip  # noqa: E402

Z = 3.0
VW, VH = int(pv.W * Z), int(pv.H * Z)
GRID_W, GRID_H = 360, 270


def load(path):
    import torch

    m = build_model()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval().to(_default_device())


def well_map(scores: np.ndarray, true: list[int], title: str, norm: float) -> np.ndarray:
    img = np.full((GRID_H, GRID_W, 3), 24, np.uint8)
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    sp = 26
    x0, y0 = 22, 42
    v = np.clip(scores / max(norm, 1e-6), 0, 1)
    col = cv2.applyColorMap((v * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_INFERNO).reshape(-1, 3)
    for k in range(96):
        c = (x0 + (k % 12) * sp + sp // 2, y0 + (k // 12) * sp + sp // 2)
        cv2.circle(img, c, 11, tuple(int(x) for x in col[k]), -1, cv2.LINE_AA)
        if k in true:
            cv2.circle(img, c, 13, (0, 255, 0), 2, cv2.LINE_AA)
    best = int(np.argmax(scores))
    cb = (x0 + (best % 12) * sp + sp // 2, y0 + (best // 12) * sp + sp // 2)
    cv2.drawMarker(img, cb, (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
    return img


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="Plate_4_clip_0003,Plate_4_clip_0009,Plate_4_clip_0017,Plate_9_clip_0019")
    ap.add_argument("--b", default=str(REPO_ROOT / "models" / "tip_b" / "best.pt"))
    ap.add_argument("--c", default=str(REPO_ROOT / "models" / "tip_c" / "best.pt"))
    ap.add_argument("--fps", type=float, default=10)
    a = ap.parse_args()
    test = set(load_splits()["test"])
    clips = [c for c in a.clips.split(",") if c]
    if any(plate_of(c) in test for c in clips):
        sys.exit("test plate clip requested; refusing.")
    dev = _default_device()
    mb, mc = load(a.b), (load(a.c) if Path(a.c).exists() else None)
    idx = torch.from_numpy(cell_index()).to(dev)
    wells = clip_wells()
    run = new_run_dir("tip-heatmaps")
    W, H = VW + GRID_W, max(VH, 2 * GRID_H) + 40
    vw = cv2.VideoWriter(str(run / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    for clip in clips:
        d = load_clip(clip)
        true = wells[clip]
        acc = np.zeros(96)
        for i in np.flatnonzero(d["has_grid"]):
            v = d["views"][i]
            with torch.no_grad():
                x = to_tensor([v]).to(dev)
                hm = torch.sigmoid(mb(x))[0, 0].cpu().numpy()
                cs = None if mc is None else well_scores(mc, x, idx)[0].cpu().numpy()
            P, S = peaks(hm, 0.3)
            img = cv2.resize(v, (VW, VH), interpolation=cv2.INTER_CUBIC)
            col = cv2.resize(cv2.applyColorMap((np.clip(hm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                             (VW, VH))  # fmt: skip
            al = cv2.resize(np.clip(hm, 0, 1), (VW, VH))[..., None] * 0.6
            img = (img * (1 - al) + col * al).astype(np.uint8)
            for k in true:
                cv2.circle(img, tuple((pv.NODES[k] * Z).astype(int)), int(0.5 * pv.PX * Z), (0, 255, 0), 1)
            if len(P):
                j = int(np.argmax(S))
                cv2.drawMarker(img, tuple((P[j] * Z).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                k = pv.nearest_well(P[j])
                if k is not None:
                    acc[k] += float(S[j])
            canvas = np.zeros((H, W, 3), np.uint8)
            canvas[40 : 40 + VH, :VW] = img
            pred = well_str(index_well(int(np.argmax(acc)))) if acc.max() > 0 else "-"
            canvas[40 : 40 + GRID_H, VW:] = well_map(acc, true, f"B: summed so far -> {pred}", acc.max())
            if cs is not None:
                canvas[40 + GRID_H : 40 + 2 * GRID_H, VW:] = well_map(
                    np.maximum(cs, 0), true, "C: this frame", max(cs.max(), 1)
                )
            head = (f"{clip} f{i}  B peak {hm.max():.2f}  true {', '.join(well_str(index_well(k)) for k in true)}"
                    f"  red = B peak, green = dispensed")  # fmt: skip
            cv2.putText(canvas, head, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(canvas)
        print(f"{clip}: B decision {well_str(index_well(int(np.argmax(acc))))}, true "
              f"{[well_str(index_well(k)) for k in true]}", flush=True)  # fmt: skip
    vw.release()
    encode(run / "raw.mp4", run / "tip_heatmaps.mp4")
    (run / "notes.md").write_text(f"# Tip heatmaps\n\nB weights {a.b}; C weights {a.c}. Clips: {clips}. "
                                  "Contains dataset frames: keep local.\n")  # fmt: skip
    print("->", run / "tip_heatmaps.mp4")


if __name__ == "__main__":
    main()
