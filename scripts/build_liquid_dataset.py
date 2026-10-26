"""Liquid-release training data from the SPACE clips (scripts/build_liquid_clips.py): every 2nd frame of each clip,
the plate warped to the well grid (pipette_coach/liquid_net.py), and the press's well: from the fill order
(--labels order, pipette8), or from the tip model on the corrected grid (--labels tip: the well under the tip end in
the frames within --tip-window s of the press, majority vote; for recordings in no set order, e.g. pipette9).

Grid per frame: the live PlateTracker run over the clip. Its grid can land a whole well off at a rebuild (about once
a second) and jump back at the next, which would put the label on the wrong well. So each clip is cut into steady
stretches (no jump of more than --max-dev well pitches), each stretch is tested at its own place and one well off in
each direction (snap_scores: well-like cells inside the 8 x 12 block, plate rim / mat just outside it), and the clip
keeps the corrected grid with the most evidence; frames whose corrected grid disagrees with it are left out, and so
is a clip with too little evidence (--min-margin).
Output: data/processed/liquid_ds/<rec>/plates.npy (N, 256, 384, 3) uint8 and meta.json (per sample: press k, frame in
the clip, video frame, seconds from the press, well (col, row), grid deviation; plus dropped clips).

    python scripts/build_liquid_dataset.py --rec pipette8
    python scripts/build_liquid_dataset.py --rec pipette9 --labels tip
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.liquid_net import S, press_well, warp_plate  # noqa: E402

CENTRES = np.array([[c, r, 1.0] for r in range(8) for c in range(12)]).T  # (3, 96)
M = 2  # border cells warped around the plate for the snap test
_yy, _xx = np.mgrid[:S, :S] - (S - 1) / 2
_RR = np.hypot(_xx, _yy) / S
RING, CORNER = (_RR > 0.28) & (_RR < 0.5), _RR > 0.62  # well rim; the plastic between wells


def well_px(H: np.ndarray) -> np.ndarray:
    p = H @ CENTRES
    return (p[:2] / p[2]).T  # (96, 2)


def shift(H: np.ndarray, dc: int, dr: int) -> np.ndarray:
    """The grid moved by (dc, dr) wells."""
    return H @ np.array([[1, 0, dc], [0, 1, dr], [0, 0, 1]], np.float64)


def wellness(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """(8 + 2M, 12 + 2M) per cell around and on the grid: correlation with this frame's typical well (median of the
    middle cells, which are wells even if the grid is one off). High on a well, low on the plate rim or the mat."""
    A = np.array([[S, 0, S / 2 + M * S], [0, S, S / 2 + M * S], [0, 0, 1]], np.float64)
    w = cv2.warpPerspective(frame, A @ np.linalg.inv(H), ((12 + 2 * M) * S, (8 + 2 * M) * S))
    L = cv2.cvtColor(w, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    c = L.reshape(8 + 2 * M, S, 12 + 2 * M, S).transpose(0, 2, 1, 3).reshape(8 + 2 * M, 12 + 2 * M, S * S)
    c = (c - c.mean(-1, keepdims=True)) / (c.std(-1, keepdims=True) + 1e-3)
    t = np.median(c[M + 2 : M + 6, M + 3 : M + 9].reshape(-1, S * S), 0)
    t = (t - t.mean()) / (t.std() + 1e-3)
    return (c * t).mean(-1)


def snap_scores(R: np.ndarray) -> dict:
    """{(dc, dr): score} for the grid moved by -1..1 wells: the weakest edge row / column inside the 8 x 12 block
    minus the strongest row / column just outside it (medians, so a hand or tip over a few cells barely counts)."""
    out = {}
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, c0 = M + dr, M + dc
            inside = [np.median(R[r0 + i, c0 : c0 + 12]) for i in (0, 7)] + [
                np.median(R[r0 : r0 + 8, c0 + j]) for j in (0, 11)
            ]
            outside = [np.median(R[r0 - 1, c0:c0 + 12]), np.median(R[r0 + 8, c0:c0 + 12]),
                       np.median(R[r0:r0 + 8, c0 - 1]), np.median(R[r0:r0 + 8, c0 + 12])]  # fmt: skip
            out[(dc, dr)] = float(min(inside) - max(outside))
    return out


def stretches(Hs: list, max_dev: float) -> list[list[int]]:
    """Runs of consecutive frames with a grid and no jump bigger than max_dev well pitches between neighbours."""
    runs, prev = [], None
    for i, H in enumerate(Hs):
        if H is None:
            prev = None
            continue
        p = well_px(H)
        if prev is None or np.linalg.norm(p - prev, axis=1).max() / np.linalg.norm(p[1] - p[0]) > max_dev:
            runs.append([])
        runs[-1].append(i)
        prev = p
    return runs


def settle_grids(frames: list, Hs: list, max_dev: float, min_margin: float, every: int = 3) -> list:
    """Per frame the corrected grid, or None (no grid, or it disagrees with the clip's best-supported grid)."""
    cands = []  # (stretch, (dc, dr), evidence)
    for run in stretches(Hs, max_dev):
        tot: dict = {}
        for i in run[::every]:
            for key, v in snap_scores(wellness(frames[i], Hs[i])).items():
                tot[key] = tot.get(key, 0.0) + v
        best = sorted(tot.items(), key=lambda kv: -kv[1])
        cands.append((run, best[0][0], best[0][1] - best[1][1]))  # evidence: lead over the runner-up
    if not cands:
        return [None] * len(Hs)
    a1 = [well_px(shift(Hs[run[0]], *off))[0] for run, off, _ in cands]
    pitch = np.linalg.norm(well_px(Hs[cands[0][0][0]])[1] - well_px(Hs[cands[0][0][0]])[0])
    evid = [ev for _, _, ev in cands]
    support = [sum(e for e, q in zip(evid, a1, strict=True) if np.linalg.norm(q - p) < 0.5 * pitch) for p in a1]
    j = int(np.argmax(support))
    out = [None] * len(Hs)
    if support[j] < min_margin:
        return out
    for (run, off, _), q in zip(cands, a1, strict=True):
        if np.linalg.norm(q - a1[j]) < 0.5 * pitch:
            for i in run:
                out[i] = shift(Hs[i], *off)
    return out


def grid_deviation(Hs: list) -> np.ndarray:
    """Per frame: largest distance of a well centre from the clip's median position, in well pitches (inf: no grid)."""
    ok = [i for i, H in enumerate(Hs) if H is not None]
    dev = np.full(len(Hs), np.inf)
    if not ok:
        return dev
    pts = np.stack([well_px(Hs[i]) for i in ok])
    med = np.median(pts, 0)
    pitch = np.median(np.linalg.norm(med.reshape(8, 12, 2)[:, 1:] - med.reshape(8, 12, 2)[:, :-1], axis=-1))
    dev[ok] = np.linalg.norm(pts - med, axis=-1).max(1) / pitch
    return dev


def tip_votes(frames: list, Gs: list, idxs: list[int], finder) -> dict:
    """Well under the tip end in each frame of idxs (grid Gs[i]) -> majority well, its share of the frames with a
    tip on a well, how many frames had one, and all votes."""
    corners = np.array([[-1.2, -1.2, 1], [12.2, -1.2, 1], [12.2, 8.2, 1], [-1.2, 8.2, 1]]).T
    votes: dict = {}
    for i in idxs:
        p = Gs[i] @ corners
        _, well = finder(frames[i], {"outline": (p[:2] / p[2]).T, "H": Gs[i]})
        if well is not None:
            votes[tuple(well)] = votes.get(tuple(well), 0) + 1
    n = sum(votes.values())
    if not n:
        return {"well": None, "share": 0.0, "n": 0, "votes": {}}
    best = max(votes, key=votes.get)
    return {"well": best, "share": round(votes[best] / n, 2), "n": n,
            "votes": {f"{c},{r}": v for (c, r), v in votes.items()}}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="pipette8")
    ap.add_argument("--step", type=int, default=2, help="keep every n-th frame")
    ap.add_argument("--max-dev", type=float, default=0.35, help="grid jump that starts a new stretch (well pitches)")
    ap.add_argument("--min-margin", type=float, default=1.0, help="least snap evidence to keep a clip")
    ap.add_argument("--reuse-grids", action="store_true", help="use the tracker grids cached by an earlier run")
    ap.add_argument("--labels", choices=["order", "tip"], default="order")
    ap.add_argument("--tip-model", default="models/tip_colors/weights/best.pt")
    ap.add_argument("--tip-window", type=float, default=0.4, help="seconds around the press the tip votes in")
    a = ap.parse_args()
    finder = None
    if a.labels == "tip":
        from ultralytics import YOLO

        from pipette_coach.live_pipeline import TipFinder

        finder = TipFinder(YOLO(str(REPO_ROOT / a.tip_model)), imgsz=640)
    from pipette_coach.localizer.detector import PlateDetector
    from pipette_coach.localizer.live import PlateTracker
    from pipette_coach.localizer.well_net import WellNet

    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    src = PROCESSED_DIR / "liquid_clips" / a.rec
    meta = json.loads((src / "clips.json").read_text())
    clips, fps = meta["clips"], meta["fps"]
    out = PROCESSED_DIR / "liquid_ds" / a.rec
    out.mkdir(parents=True, exist_ok=True)
    cap_n = sum((c["f1"] - c["f0"]) // a.step + 1 for c in clips)
    plates = np.lib.format.open_memmap(out / "plates.npy", "w+", np.uint8, (cap_n, 8 * S, 12 * S, 3))
    samples, dropped, labels, n = [], [], [], 0
    cache = out / "grids.npz"
    cached = dict(np.load(cache)) if a.reuse_grids and cache.exists() else {}
    for c in clips:
        tr = PlateTracker(cfg, det, net)
        cap = cv2.VideoCapture(str(src / f"mark_{c['k']:03d}.mp4"))
        frames, Hs = [], []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
            if f"H{c['k']}" not in cached:
                Hs.append(tr.update(f).get("H"))
        if f"H{c['k']}" in cached:
            Hs = [None if np.isnan(H).any() else H for H in cached[f"H{c['k']}"]]
        cached[f"H{c['k']}"] = np.stack([np.full((3, 3), np.nan) if H is None else H for H in Hs])
        Gs = settle_grids(frames, Hs, a.max_dev, a.min_margin)
        dev = grid_deviation(Gs)
        keep = [i for i in range(0, len(frames), a.step) if Gs[i] is not None]
        if not keep:
            dropped.append({"k": c["k"], "why": "grid not settled"})
            continue
        if finder is None:
            col, row = press_well(c["k"])
        else:
            win = [i for i in range(len(frames)) if Gs[i] is not None
                   and abs(c["f0"] + i - c["mark_frame"]) <= a.tip_window * fps]  # fmt: skip
            got = tip_votes(frames, Gs, win, finder)
            if not got["well"]:
                dropped.append({"k": c["k"], "why": "no tip on a well near the press"})
                continue
            col, row = got["well"]
            labels.append({"k": c["k"], "col": col, "row": row, **{x: got[x] for x in ("share", "n", "votes")}})
        for i in keep:
            plates[n] = warp_plate(frames[i], Gs[i])
            samples.append({"idx": n, "k": c["k"], "i": i, "frame": c["f0"] + i,
                            "dt": round((c["f0"] + i - c["mark_frame"]) / fps, 3), "col": col, "row": row,
                            "dev": round(float(dev[i]), 3)})  # fmt: skip
            n += 1
        lost = (len(frames) + a.step - 1) // a.step - len(keep)
        if lost:
            dropped.append({"k": c["k"], "why": f"{lost} frames with no or a disagreeing grid"})
        print(f"press {c['k']:2d}: {len(keep)} frames kept, {lost} left out", flush=True)
    plates.flush()
    np.savez(cache, **cached)
    (out / "meta.json").write_text(json.dumps({"rec": a.rec, "fps": fps, "step": a.step, "max_dev": a.max_dev,
                                               "n": n, "labels_from": a.labels, "tip_labels": labels,
                                               "samples": samples, "dropped": dropped}, indent=0))  # fmt: skip
    print(f"{n} samples from {len({s['k'] for s in samples})} presses -> {out}")


if __name__ == "__main__":
    main()
