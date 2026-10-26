"""K-fold for the before/after liquid model on the crop around the tip's well (scripts/train_well_compare.py's data,
pairs and model), pipette8 only, all presses (no held-out test, at the user's request).

Folds: --folds blocks of consecutive presses (one plate, so blocks in time). Fold f validates on pairs with both views
inside its block and trains on pairs entirely before or entirely after it (no pair spans the block, so the model
never sees the block's wells change). A fixed number of epochs (no best-epoch pick on the fold's own val), so the
pooled val numbers are honest for pipette8. The rule (well_change.py thresholds) is scored on the same val pairs.
Last, one model on all pairs -> models/<name>/final.pt (live: live_demo --liquid compare --compare-model ...).

    python scripts/train_compare_kfold.py
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_well_compare import auc, crop_pair, liquid_of, make_pairs, predict, report, rule_scores  # noqa: E402

from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.liquid_net import LiquidCropNet  # noqa: E402


def fold_pairs(clips, dev, lo: int, hi: int, n: int, gap: int):
    """(train, val) pairs for the block of presses [lo, hi)."""
    return make_pairs(clips, dev, 0, lo, gap) + make_pairs(clips, dev, hi, n, gap), make_pairs(clips, dev, lo, hi, gap)


def train_model(views, pairs, a, rng, dev, log=None):
    pos = [p for p in pairs if p[4] == 1]
    hard = [p for p in pairs if p[5] == "tip_near"]
    easy = [p for p in pairs if p[4] == 0 and p[5] != "tip_near"]
    model = LiquidCropNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n_neg = len(pos) * a.neg_per_pos
    steps = a.epochs * ((len(pos) + n_neg) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(1, a.epochs + 1):
        model.train()
        nh = min(len(hard), n_neg // 2)
        items = pos + rng.sample(hard, nh) + rng.sample(easy, n_neg - nh)
        rng.shuffle(items)
        t0, tot = time.time(), 0.0
        for b in range(0, len(items) - a.bs + 1, a.bs):
            chunk = items[b : b + a.bs]
            X = torch.from_numpy(np.stack([crop_pair(views, p, rng) for p in chunk])).to(dev)
            Y = torch.tensor([p[4] for p in chunk], dtype=torch.float32, device=dev)
            loss = lossf(model(X), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(chunk)
        if log:
            log(f"  epoch {ep}: loss {tot / len(items):.4f} ({time.time() - t0:.0f}s)")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="pipette8")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--gap", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--name", default="well_compare_cv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = PROCESSED_DIR / "well_views" / a.rec
    clips = json.loads((d / "meta.json").read_text())["clips"]
    views, wdev = np.load(d / "plates.npy"), np.load(d / "dev.npy")
    n = 96
    liquid = liquid_of(views, clips)
    run = REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{a.name}"
    run.mkdir(parents=True)
    print(f"liquid per column {liquid}\nrun {run}", flush=True)
    size = n // a.folds
    folds, all_val, all_p, all_rule = [], [], [], []
    for f in range(a.folds):
        lo, hi = f * size, (f + 1) * size if f < a.folds - 1 else n
        train, val = fold_pairs(clips, wdev, lo, hi, n, a.gap)
        print(f"fold {f}: presses {lo}-{hi - 1}, train {len(train)} pairs ({sum(p[4] for p in train):.0f} added), "
              f"val {len(val)} ({sum(p[4] for p in val):.0f} added)", flush=True)  # fmt: skip
        model = train_model(views, train, a, rng, dev, log=lambda s: print(s, flush=True))
        p, r = predict(model, views, val, dev), rule_scores(views, val)
        y = np.array([q[4] for q in val])
        m = {"fold": f, "presses": [lo, hi - 1], "auc": round(auc(p, y), 4), "model": report(p, val, liquid),
             "rule": report(r, val, liquid)}  # fmt: skip
        print(m, flush=True)
        folds.append(m)
        all_val += val
        all_p.append(p)
        all_rule.append(r)
    p, r = np.concatenate(all_p), np.concatenate(all_rule)
    y = np.array([q[4] for q in all_val])
    pooled = {"auc": round(auc(p, y), 4), "model": {thr: report(p, all_val, liquid, thr) for thr in (0.3, 0.5, 0.8)},
              "rule": report(r, all_val, liquid)}  # fmt: skip
    print("POOLED:", json.dumps(pooled, indent=1), flush=True)
    print("final model on all pairs", flush=True)
    model = train_model(views, make_pairs(clips, wdev, 0, n, a.gap), a, rng, dev)
    out = REPO_ROOT / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(a), "cv": pooled}, out / "final.pt")
    (run / "metrics.json").write_text(json.dumps({"rec": a.rec, "args": vars(a), "liquid_per_column": liquid,
                                                  "folds": folds, "pooled": pooled}, indent=1))  # fmt: skip
    print("done ->", out / "final.pt", run, flush=True)


if __name__ == "__main__":
    main()
