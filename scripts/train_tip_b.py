"""Model B: learned pipette-tip keypoint (heatmap U-Net) in the plate view.

Labels (no hand clicks): single-channel train clips (plates 9, 10). A frame is a positive when the geometric tip
(shaft end, tracking/tip_geom.py) lies within labels_b.near_pitch of the clip's labelled well (labels.json); the
target is a Gaussian at that WELL centre (the clear tip end sits in the well, past the visible shaft end). Every
other pixel of a positive frame is negative (shaft, hand, other wells). Frames with no foreground in the well area
are all-negative. Other frames (pipette travelling, tip end unknown) are not used.

Val (plate 4, single wells): per frame the heatmap peak -> nearest well; clip prediction = well with the largest
summed peak score over frames. Metrics: exact accuracy, neighbour (off by one) rate.

    python scripts/train_tip_b.py
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
from pipette_coach.localizer.detector import _default_device  # noqa: E402
from pipette_coach.localizer.well_net import build_model, peaks, to_tensor  # noqa: E402
from pipette_coach.protocol.wells import index_well, well_str  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tracking import plate_view as pv  # noqa: E402
from pipette_coach.tracking.tip_data import clip_wells, load_clip, split_clips  # noqa: E402

SIGMA = 2.0  # px in the plate view (0.14 well spacing)


def target(pts: np.ndarray | None) -> np.ndarray:
    t = np.zeros((pv.H, pv.W), np.float32)
    if pts is not None:
        yy, xx = np.mgrid[0 : pv.H, 0 : pv.W].astype(np.float32)
        for x, y in np.atleast_2d(pts):
            t = np.maximum(t, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA**2)).astype(np.float32))
    return t


def build_items(clips: list[str], wells: dict, lc: dict) -> list[tuple[np.ndarray, np.ndarray | None]]:
    items = []
    for c in clips:
        d = load_clip(c)
        k = wells[c][0]
        dist = np.linalg.norm(d["tip"] - pv.NODES[k], axis=1) / pv.PX
        for i in np.flatnonzero(d["has_grid"]):
            if np.isfinite(dist[i]) and dist[i] <= lc["near_pitch"]:
                items.append((d["views"][i], pv.NODES[k].copy()))
            elif not np.isfinite(d["tip"][i, 0]) and d["fg_area"][i] < lc["max_empty_fg"]:
                items.append((d["views"][i], None))
    return items


def build_items_clicks(path: Path) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Hand-clicked tips (scripts/label_tips.py): 'tip' frames -> all clicked tips (multichannel: every tip),
    'none' frames -> all-negative. Train clips only (the tool refuses val/test)."""
    import json

    lab = json.loads(path.read_text())["labels"]
    items = []
    for c, frames in lab.items():
        d = load_clip(c)
        for f, e in frames.items():
            v = d["views"][int(f)]
            items.append((v, np.array(e["tips_view"], float) if e["kind"] == "tip" else None))
    return items


def augment(img: np.ndarray, pt: np.ndarray | None, rng: np.random.Generator):
    dx, dy = rng.integers(-4, 5, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (pv.W, pv.H), borderMode=cv2.BORDER_REPLICATE)
    pt = None if pt is None else np.atleast_2d(pt) + [dx, dy]
    if rng.random() < 0.5:  # pipettes come from any side: flip the view (and the tips) both ways
        img = img[:, ::-1]
        pt = None if pt is None else np.c_[pv.W - 1 - pt[:, 0], pt[:, 1]]
    if rng.random() < 0.5:
        img = img[::-1]
        pt = None if pt is None else np.c_[pt[:, 0], pv.H - 1 - pt[:, 1]]
    img = img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    img = img + rng.normal(0, rng.uniform(0, 4), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), pt


def clip_scores(model, dev, views: np.ndarray, has: np.ndarray, thr: float) -> np.ndarray:
    """(96,) summed per-frame peak score per well."""
    import torch

    s = np.zeros(96)
    idx = np.flatnonzero(has)
    with torch.no_grad():
        for b in range(0, len(idx), 16):
            x = to_tensor([views[i] for i in idx[b : b + 16]]).to(dev)
            prob = torch.sigmoid(model(x))[:, 0].cpu().numpy()
            for p in prob:
                P, S = peaks(p, thr)
                if len(P):
                    j = int(np.argmax(S))
                    k = pv.nearest_well(P[j])
                    if k is not None:
                        s[k] += float(S[j])
    return s


def neighbours(a: int, b: int) -> bool:
    return max(abs(a // 12 - b // 12), abs(a % 12 - b % 12)) == 1


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--labels",
        choices=["auto", "clicks"],
        default="auto",
        help="auto: geometric bootstrap filtered by labels.json; clicks: data/annotations/tip_clicks.json",
    )
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    lc = load_config("tracking")["labels_b"]
    wells = clip_wells()
    if a.labels == "clicks":
        items = build_items_clicks(REPO_ROOT / "data" / "annotations" / "tip_clicks.json")
        train = sorted({c for c in json.loads((REPO_ROOT / "data" / "annotations" / "tip_clicks.json").read_text())[
            "labels"]})  # fmt: skip
    else:
        train = split_clips("train", {9, 10})
        items = build_items(train, wells, lc)
    n_pos = sum(p is not None for _, p in items)
    print(f"train: {len(train)} clips, {n_pos} positive / {len(items) - n_pos} empty frames", flush=True)
    dev = _default_device()
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = max(1, len(items) // 16)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * steps)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for _ in range(steps):
            batch = [augment(*items[i], rng) for i in rng.integers(0, len(items), 16)]
            x = to_tensor([b[0] for b in batch]).to(dev)
            t = torch.from_numpy(np.stack([target(b[1]) for b in batch])[:, None]).to(dev)
            pr = torch.sigmoid(model(x))
            w = 1 + 50 * t  # one tip per frame: weight the positive blob up
            loss = (w * (pr - t) ** 2).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep + 1}/{a.epochs} loss {tot / steps:.5f} ({time.time() - t0:.0f}s)", flush=True)
    out = REPO_ROOT / "models" / ("tip_b" if a.labels == "auto" else "tip_b_clicks")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "best.pt")

    # ---- val (plate 4): clip-level well prediction
    model.eval()
    rows, ok, nb = [], 0, 0
    val = split_clips("val")
    t1 = time.perf_counter()
    n_frames = 0
    for c in val:
        d = load_clip(c)
        n_frames += int(d["has_grid"].sum())
        s = clip_scores(model, dev, d["views"], d["has_grid"], 0.3)
        pred, true = int(np.argmax(s)), wells[c][0]
        hit = s.max() > 0 and pred == true
        ok += hit
        nb += (not hit) and neighbours(pred, true)
        rows.append((c, well_str(index_well(true)), well_str(index_well(pred)) if s.max() > 0 else "-", hit))
    ms = (time.perf_counter() - t1) * 1000 / max(n_frames, 1)
    run = new_run_dir(f"tip-B-{a.labels}")
    metrics = {"model": f"B tip keypoint ({a.labels} labels)", "train_clips": len(train), "train_pos_frames": n_pos,
               "val_clips": len(val), "val_exact_accuracy": ok / len(val), "val_neighbour_rate": nb / len(val),
               "ms_per_frame": ms}  # fmt: skip
    write_metrics(run, metrics)
    lines = ["# Tip model B (learned keypoint)", "", f"Val plate 4: exact {ok}/{len(val)}, off-by-one {nb}. "
             f"{ms:.1f} ms/frame. Labels: geometric shaft end near the labelled well -> Gaussian at that well.", "",
             "| clip | true | predicted | ok |", "|---|---|---|---|"]  # fmt: skip
    lines += [f"| {c} | {t} | {p} | {'yes' if h else 'no'} |" for c, t, p, h in rows]
    lines += ["", "Regenerate: `python scripts/train_tip_b.py` (seed 0) -> models/tip_b/best.pt."]
    (run / "notes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
