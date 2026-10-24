"""Train the learned grid verifier (pipette_coach/localizer/verifier.py).

Positives: frames of UNFLAGGED (or eye-verified, configs/verified_grids.json) train clips rectified with their
grid (data/processed/clip_grids.json), with small
jitter (+-0.08 pitch, +-1 deg, +-1% scale) so it tolerates slightly-off grids. Negatives: the same frames with
the grid off by half a well, one row/column, or a random 0.3-1.2 pitch offset / rotation / scale.
Val (plate 4): for each frame, 9 candidates (true grid + half-well and one-row/column shifts, each with small
jitter) -> is the true one ranked first? Compared with the hand-made baseline (mean radial-edge score).

    python scripts/clip_grids.py && python scripts/train_grid_verifier.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, load_splits, plate_of  # noqa: E402
from pipette_coach.localizer.geometry import GRID, project  # noqa: E402
from pipette_coach.localizer.rings import radial_edge_response  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.localizer.verifier import OFFSETS, build_model, offset_H, rectify, to_tensor  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402


def jitter(rng: np.random.Generator, H: np.ndarray, small: bool = True) -> np.ndarray:
    if small:
        return offset_H(H, *rng.uniform(-0.08, 0.08, 2), rng.uniform(-1, 1), 1 + rng.uniform(-0.01, 0.01))
    while True:  # a clearly wrong grid: >= 0.3 pitch off somewhere
        d = rng.uniform(-1.2, 1.2, 2)
        if np.hypot(*d) >= 0.3:
            return offset_H(H, *d, rng.uniform(-3, 3), 1 + rng.uniform(-0.04, 0.04))


def make_set(clips: dict, rng: np.random.Generator, every: int, n_neg: int) -> tuple[list, list]:
    X, y = [], []
    for clip, info in clips.items():
        H = np.array(info["H"])
        for f in read_frames(clip, every):
            X.append(rectify(f, jitter(rng, H)))
            y.append(1.0)
            for k in range(n_neg):
                if k % 2 == 0:  # the structured confusions: half well, one row/column
                    dx, dy = OFFSETS[1 + rng.integers(len(OFFSETS) - 1)]
                    Hn = offset_H(jitter(rng, H), dx, dy)
                else:
                    Hn = jitter(rng, H, small=False)
                X.append(rectify(f, Hn))
                y.append(0.0)
    return X, y


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.5:
        out = out[:, ::-1]  # the plate grid is symmetric: flips keep "aligned" aligned
    if rng.random() < 0.5:
        out = out[::-1]
    return np.clip(out, 0, 255).astype(np.uint8)


def baseline_scores(frame: np.ndarray, cands: list[np.ndarray], cfg: dict) -> np.ndarray:
    """Hand-made selector: mean radial-edge score at the 96 well positions (one shared map)."""
    pts = np.concatenate([project(H, GRID) for H in cands])
    x0, y0 = np.floor(pts.min(0) - 12).astype(int)
    x1, y1 = np.ceil(pts.max(0) + 12).astype(int)
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    r, _ = radial_edge_response(frame[y0:y1, x0:x1], cfg)
    out = []
    for H in cands:
        q = project(H, GRID) - [x0, y0]
        v = cv2.remap(r, q[:, :1].astype(np.float32), q[:, 1:].astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
        out.append(float(v.mean()))
    return np.array(out)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--neg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="configs/splits.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    cfg = load_config("localizer")
    splits = load_splits(a.splits)
    grids = json.loads((PROCESSED_DIR / "clip_grids.json").read_text())
    verified = set(json.loads((REPO_ROOT / "configs" / "verified_grids.json").read_text())) - {"_why"}
    usable = {c: g for c, g in grids.items() if g["H"] is not None and (not g["flags"] or c in verified)}
    train = {c: g for c, g in usable.items() if plate_of(c) in splits["train"]}
    val = {c: g for c, g in usable.items() if plate_of(c) in splits["val"]}
    print(f"train clips {len(train)} (of {sum(plate_of(c) in splits['train'] for c in grids)}), val clips {len(val)}")
    X, y = make_set(train, rng, a.every, a.neg)
    print(f"train samples {len(X)} ({int(sum(y))} positive)")

    from pipette_coach.localizer.detector import _default_device

    dev = _default_device()
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    yt = np.array(y, np.float32)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for b in range(0, len(perm), 64):
            idx = perm[b : b + 64]
            xb = to_tensor([augment(X[i], rng) for i in idx]).to(dev)
            yb = torch.from_numpy(yt[idx]).to(dev)
            loss = loss_fn(model(xb).squeeze(1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(X):.4f} ({time.time() - t0:.0f}s)", flush=True)

    # val: rank the true grid among 9 candidates (each candidate slightly jittered, like a real pipeline grid)
    model.eval()
    hits, hits_base, margins, n = 0, 0, [], 0
    with torch.no_grad():
        for clip, info in val.items():
            H = np.array(info["H"])
            for f in read_frames(clip, 4):
                cands = [offset_H(jitter(rng, H), dx, dy) for dx, dy in OFFSETS]
                s = torch.sigmoid(model(to_tensor([rectify(f, Hc) for Hc in cands]).to(dev))).cpu().numpy().ravel()
                hits += int(np.argmax(s) == 0)
                margins.append(float(s[0] - np.max(s[1:])))
                hits_base += int(np.argmax(baseline_scores(f, cands, cfg)) == 0)
                n += 1
    out = REPO_ROOT / "models" / "grid_verifier"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "best.pt")
    run = new_run_dir("grid-verifier")
    metrics = {
        "train_clips": len(train),
        "val_clips": len(val),
        "val_frames": n,
        "val_top1_learned": hits / max(n, 1),
        "val_top1_radial_edge_baseline": hits_base / max(n, 1),
        "val_margin_median": float(np.median(margins)) if margins else None,
        "epochs": a.epochs,
        "seed": a.seed,
    }
    write_metrics(run, metrics)
    (run / "notes.md").write_text(
        "# Grid verifier\n\nResNet-18 on rectified plate views; picks the true grid among 9 candidates (true + "
        "half-well / one-row / one-column shifts, each jittered).\n\n"
        f"- Val top-1: learned {metrics['val_top1_learned']:.1%} vs radial-edge baseline "
        f"{metrics['val_top1_radial_edge_baseline']:.1%} ({n} frames, plate 4)\n"
        f"- Train: {len(train)} unflagged train clips (labels = pipeline grids that passed the review flags)\n\n"
        "Regenerate: `python scripts/clip_grids.py && python scripts/train_grid_verifier.py`. Weights: "
        "models/grid_verifier/best.pt\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
