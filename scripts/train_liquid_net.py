"""Fine-tune the liquid-release net (pipette_coach/liquid_net.py, ImageNet ResNet18) on SPACE-marked dispenses.

Data: scripts/build_liquid_dataset.py. Label: the press's well is "liquid going in" for every frame of its clip
(SPACE +- 1 s); all other wells are not. Split by press (one plate only, so a time split): --train-to / --val-to.
The test presses are never loaded here. Augmentation: flips (labels follow), hue rotation (other liquid colours),
brightness / contrast, a few px of shift (grid error).
Output: models/<name>/best.pt and runs/<timestamp>_<name>/ (metrics.json, notes.md).

    python scripts/train_liquid_net.py --rec pipette8
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.liquid_net import DT, LiquidNet, flip, shift_hue, to_input  # noqa: E402


def pairs(samples: list[dict], step: int) -> list[tuple[dict, int]]:
    """(sample, index of the frame ~DT earlier in the same clip); samples without one are skipped."""
    at = {(s["k"], s["i"]): s["idx"] for s in samples}
    out = []
    for s in samples:
        for back in (DT, DT + step, DT - step):
            j = at.get((s["k"], s["i"] - back))
            if j is not None:
                out.append((s, j))
                break
    return out


def augment(now, before, target, rng: random.Random):
    now, before, target = flip(now, before, target, rng.random() < 0.5, rng.random() < 0.5)
    if rng.random() < 0.7:
        dh = rng.randrange(180)
        now, before = shift_hue(now, dh), shift_hue(before, dh)
    a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
    dx, dy = rng.randint(-4, 4), rng.randint(-4, 4)
    T = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = now.shape[:2]
    now, before = (cv2.warpAffine(cv2.convertScaleAbs(im, alpha=a, beta=b), T, (w, h), borderMode=cv2.BORDER_REFLECT)
                   for im in (now, before))  # fmt: skip
    return now, before, target


def batch(P, items, rng=None):
    X, Y = [], []
    for s, j in items:
        now, before = np.array(P[s["idx"]]), np.array(P[j])
        t = np.zeros((8, 12), np.float32)
        t[s["row"], s["col"]] = 1
        if rng is not None:
            now, before, t = augment(now, before, t, rng)
        X.append(to_input(now, before))
        Y.append(t)
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


@torch.no_grad()
def evaluate(model, P, items, dev, bs=32) -> dict:
    model.eval()
    logits, wells, ks = [], [], []
    for b in range(0, len(items), bs):
        X, _ = batch(P, items[b : b + bs])
        logits.append(model(X.to(dev)).float().cpu())
        wells += [s["row"] * 12 + s["col"] for s, _ in items[b : b + bs]]
        ks += [s["k"] for s, _ in items[b : b + bs]]
    L = torch.cat(logits).reshape(len(items), 96)
    p, wells, ks = torch.sigmoid(L).numpy(), np.array(wells), np.array(ks)
    pos = p[np.arange(len(p)), wells]
    other = p.copy()
    other[np.arange(len(p)), wells] = 0
    press = {k: int(np.argmax(L[ks == k].mean(0))) == int(wells[ks == k][0]) for k in np.unique(ks)}
    return {"frames": len(items), "presses": len(press),
            "frame_top1": round(float((p.argmax(1) == wells).mean()), 4),
            "press_top1": round(float(np.mean(list(press.values()))), 4),
            "press_wrong": [int(k) for k, ok in press.items() if not ok],
            "frame_pos_over_0.5": round(float((pos > 0.5).mean()), 4),
            "frame_other_over_0.5": round(float((other.max(1) > 0.5).mean()), 4),
            "pos_prob_median": round(float(np.median(pos)), 3),
            "other_max_prob_p95": round(float(np.percentile(other.max(1), 95)), 3)}  # fmt: skip


def bench(model, dev, n=60) -> dict:
    model.eval()
    out = {}
    for name, d in (("device", dev), ("cpu", torch.device("cpu"))):
        m = model.to(d)
        x = torch.zeros(1, 6, 256, 384, device=d)
        with torch.no_grad():
            for _ in range(5):
                m(x)
            if d.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                m(x).cpu()
        out[f"ms_per_frame_{name}" if name == "cpu" else f"ms_per_frame_{d.type}"] = round(
            (time.perf_counter() - t0) / n * 1000, 1
        )
    model.to(dev)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="pipette8")
    ap.add_argument("--train-to", type=int, default=64, help="presses < this train")
    ap.add_argument("--val-to", type=int, default=80, help="presses < this (and >= --train-to) validate")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--name", default="liquid_net")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ds = PROCESSED_DIR / "liquid_ds" / a.rec
    meta = json.loads((ds / "meta.json").read_text())
    P = np.load(ds / "plates.npy", mmap_mode="r")
    items = pairs(meta["samples"], meta["step"])
    train = [x for x in items if x[0]["k"] < a.train_to]
    val = [x for x in items if a.train_to <= x[0]["k"] < a.val_to]
    print(f"train {len(train)} frames / {len({s['k'] for s, _ in train})} presses, "
          f"val {len(val)} / {len({s['k'] for s, _ in val})}", flush=True)  # fmt: skip

    model = LiquidNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(train) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(a.pos_weight, device=dev))
    out_dir = REPO_ROOT / "models" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    run = REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{a.name}"
    run.mkdir(parents=True)
    hist, best, best_key = [], None, (-1.0, -1.0)
    base = evaluate(model, P, val, dev)
    print("epoch 0 (ImageNet weights only):", base, flush=True)
    for ep in range(1, a.epochs + 1):
        model.train()
        rng.shuffle(train)
        t0, tot = time.time(), 0.0
        for b in range(0, len(train), a.bs):
            X, Y = batch(P, train[b : b + a.bs], rng)
            loss = lossf(model(X.to(dev)), Y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(X)
        m = evaluate(model, P, val, dev)
        m.update(epoch=ep, train_loss=round(tot / len(train), 4), sec=round(time.time() - t0))
        hist.append(m)
        print(m, flush=True)
        key = (m["press_top1"], m["frame_top1"])
        if key > best_key:
            best_key, best = key, m
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val": m, "args": vars(a)}, out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev)["state_dict"])
    speed = bench(model, dev)
    metrics = {"rec": a.rec, "split": {"train_presses": [0, a.train_to - 1], "val_presses": [a.train_to, a.val_to - 1],
               "test_presses": f">= {a.val_to} (not run)"}, "args": vars(a), "val_epoch0": base, "best_val": best,
               "speed": speed, "history": hist}  # fmt: skip
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("best:", best, "\nspeed:", speed, "\nrun:", run)


if __name__ == "__main__":
    main()
