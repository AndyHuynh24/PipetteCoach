"""Before/after liquid model: did liquid go into this well between two clear views of it? (LiquidCropNet, ImageNet
ResNet18 with 6 input channels: the 3x3-well crop after + before.) Live: before = the well before the tip came,
after = once the tip has left; replaces the texture / colour thresholds of pipette_coach/well_change.py.

Data: scripts/build_well_views.py (one view per press clip, covered views dropped). Pairs of views of the same well
from clips j1 < j2 (j2 - j1 <= --gap, neither clip the well's own press): label 1 if the well's press is between them.
Negatives by kind: empty (empty -> empty), filled (filled -> filled), tip_near (no change, but the tip dispensed into
a neighbouring well in between: the live wrong-well case).
Split by press (one plate, so a time split): train pairs inside presses < --train-to, val inside [--train-to,
--val-to); pairs across the boundary are not used; test presses (>= --val-to) are never loaded.
The rule (well_change.py thresholds, relative to the plate) is scored on the same val pairs.

    python scripts/train_well_compare.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_well_views import COVER_DEV, press_of  # noqa: E402

from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.liquid_net import CROP, LiquidCropNet, S, crop_well, press_well, shift_hue, to_input  # noqa: E402
from pipette_coach.well_change import cell_features  # noqa: E402

KINDS = ("added", "empty", "filled", "tip_near")


def make_pairs(clips: list[int], dev: np.ndarray, lo: int, hi: int, gap: int) -> list[tuple]:
    """[(col, row, i1, i2, label, kind)] for clip indexes i1 < i2 with both clips in [lo, hi)."""
    out = []
    ix = [(i, k) for i, k in enumerate(clips) if lo <= k < hi]
    for r in range(8):
        for c in range(12):
            k0 = press_of(c, r)
            ok = [(i, k) for i, k in ix if k != k0 and np.isfinite(dev[i, r, c]) and dev[i, r, c] <= COVER_DEV]
            for a, (i1, k1) in enumerate(ok):
                for i2, k2 in ok[a + 1 :]:
                    if k2 - k1 > gap:
                        break
                    if k1 < k0 < k2:
                        out.append((c, r, i1, i2, 1.0, "added"))
                        continue
                    near = any(
                        max(abs(press_well(j)[0] - c), abs(press_well(j)[1] - r)) == 1 for j in range(k1, k2 + 1)
                    )
                    kind = "tip_near" if near else ("empty" if k2 < k0 else "filled")
                    out.append((c, r, i1, i2, 0.0, kind))
    return out


def rule_scores(views: np.ndarray, pairs: list[tuple], tex_ratio: float = 0.8, color_de: float = 5.0) -> np.ndarray:
    """well_change.py's verdict on each pair (texture ratio and chroma shift relative to the median of the plate's
    other wells over the same two views): 1 = liquid."""
    F = np.stack([cell_features(v, S) for v in views])  # (n, 8, 12, 7)
    out = np.zeros(len(pairs), np.float32)
    memo = {}
    for n, (c, r, i1, i2, *_) in enumerate(pairs):
        if (i1, i2) not in memo:
            tex = F[i2, ..., 0] / np.maximum(F[i1, ..., 0], 1e-3)
            sh = F[i2, ..., 2:4] - F[i1, ..., 2:4]
            memo[(i1, i2)] = (tex, sh, np.median(tex), np.median(sh.reshape(-1, 2), 0))
        tex, sh, mt, ms = memo[(i1, i2)]
        out[n] = float(tex[r, c] / mt < tex_ratio or np.linalg.norm(sh[r, c] - ms) > color_de)
    return out


def liquid_of(views: np.ndarray, clips: list[int]) -> dict:
    """Column -> 'blue' / 'pink' / 'faint' from the colour change of its wells (filled vs empty, Lab a/b)."""
    F = np.stack([cell_features(v, S) for v in views])
    K = np.array(clips)
    out = {}
    for c in range(12):
        shifts = []
        for r in range(8):
            k0 = press_of(c, r)
            if (K < k0).sum() and (K > k0).sum():
                shifts.append(np.median(F[K > k0, r, c, 2:4], 0) - np.median(F[K < k0, r, c, 2:4], 0))
        if not shifts:  # not filled yet in the loaded clips
            out[c] = "n/a"
            continue
        da, db = np.median(shifts, 0)
        out[c] = "blue" if db < -6 else "pink" if da > 3 else "faint"
    return out


def crop_pair(views, p, rng=None):
    c, r, i1, i2 = p[:4]
    after, before = crop_well(views[i2], c, r), crop_well(views[i1], c, r)
    if rng is not None:
        for ax in (0, 1):
            if rng.random() < 0.5:
                after, before = np.flip(after, ax), np.flip(before, ax)
        if rng.random() < 0.7:
            dh = rng.randrange(180)
            after, before = shift_hue(np.ascontiguousarray(after), dh), shift_hue(np.ascontiguousarray(before), dh)
        a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
        n, ims = CROP * S, []
        for im in (after, before):  # each view its own small light change and grid error
            T = np.float32([[1, 0, rng.randint(-4, 4)], [0, 1, rng.randint(-4, 4)]])
            im = cv2.convertScaleAbs(np.ascontiguousarray(im), alpha=a * rng.uniform(0.93, 1.07), beta=b)
            ims.append(cv2.warpAffine(im, T, (n, n), borderMode=cv2.BORDER_REFLECT))
        after, before = ims
    return to_input(np.ascontiguousarray(after), np.ascontiguousarray(before))


@torch.no_grad()
def predict(model, views, pairs, dev, bs=256) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, len(pairs), bs):
        X = np.stack([crop_pair(views, p) for p in pairs[b : b + bs]])
        out.append(torch.sigmoid(model(torch.from_numpy(X).to(dev))).cpu().numpy())
    return np.concatenate(out)


def report(p: np.ndarray, pairs: list[tuple], liquid: dict, thr: float = 0.5) -> dict:
    y = np.array([q[4] for q in pairs])
    kind = np.array([q[5] for q in pairs])
    liq = np.array([liquid[q[0]] for q in pairs])
    out = {}
    for lq in ("blue", "pink", "faint"):
        m = (kind == "added") & (liq == lq)
        out[f"found_{lq}"] = f"{int((p[m] > thr).sum())}/{int(m.sum())}"
    out["found_all"] = round(float((p[y == 1] > thr).mean()), 4)
    for k in KINDS[1:]:
        out[f"false_{k}"] = f"{int((p[kind == k] > thr).sum())}/{int((kind == k).sum())}"
    return out


def auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(len(p))
    pos, n_pos, n_neg = ranks[y == 1], (y == 1).sum(), (y == 0).sum()
    return float((pos.sum() - n_pos * (n_pos - 1) / 2) / max(n_pos * n_neg, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="pipette8")
    ap.add_argument("--train-to", type=int, default=64)
    ap.add_argument("--val-to", type=int, default=80)
    ap.add_argument("--gap", type=int, default=4, help="max presses between the two views")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--name", default="well_compare")
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
    keep = [i for i, k in enumerate(clips) if k < a.val_to]  # test clips never loaded into training / val
    views, wdev, clips = views[keep], wdev[keep], [clips[i] for i in keep]
    liquid = liquid_of(views, clips)
    train = make_pairs(clips, wdev, 0, a.train_to, a.gap)
    val = make_pairs(clips, wdev, a.train_to, a.val_to, a.gap)
    pos = [p for p in train if p[4] == 1]
    hard = [p for p in train if p[5] == "tip_near"]
    easy = [p for p in train if p[4] == 0 and p[5] != "tip_near"]
    print(f"liquid per column {liquid}\ntrain: {len(pos)} added, {len(hard)} tip_near, {len(easy)} other no-change;"
          f" val {len(val)} pairs ({sum(p[4] for p in val):.0f} added)", flush=True)  # fmt: skip

    yv = np.array([p[4] for p in val])
    rule = rule_scores(views, val)
    rule_rep = {"auc_like": None, **report(rule, val, liquid)}
    print("rule:", rule_rep, flush=True)

    model = LiquidCropNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    per_ep = len(pos) * (1 + a.neg_per_pos)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * ((per_ep + a.bs - 1) // a.bs),
                                                pct_start=0.15)  # fmt: skip
    lossf = torch.nn.BCEWithLogitsLoss()
    out_dir = REPO_ROOT / "models" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    run = REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{a.name}"
    run.mkdir(parents=True)
    hist, best = [], None
    for ep in range(1, a.epochs + 1):
        model.train()
        n_neg = len(pos) * a.neg_per_pos
        items = (
            pos + rng.sample(hard, min(len(hard), n_neg // 2)) + rng.sample(easy, n_neg - min(len(hard), n_neg // 2))
        )
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
        p = predict(model, views, val, dev)
        m = {"epoch": ep, "train_loss": round(tot / len(items), 4), "auc": round(auc(p, yv), 4),
             "sec": round(time.time() - t0), **report(p, val, liquid)}  # fmt: skip
        hist.append(m)
        print(m, flush=True)
        if best is None or m["auc"] > best["auc"]:
            best = m
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val": m, "args": vars(a)}, out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev)["state_dict"])
    p = predict(model, views, val, dev)
    at = {thr: report(p, val, liquid, thr) for thr in (0.3, 0.5, 0.8, 0.9)}
    (run / "metrics.json").write_text(json.dumps({"rec": a.rec, "split": {"train_presses": [0, a.train_to - 1],
        "val_presses": [a.train_to, a.val_to - 1], "test_presses": f">= {a.val_to} (not run)"}, "args": vars(a),
        "liquid_per_column": liquid, "best_val": best, "model_at_thr": at, "rule_val": rule_rep,
        "history": hist}, indent=1))  # fmt: skip
    print("best:", best, "\nat thresholds:", json.dumps(at), "\nrule:", rule_rep, "\nrun:", run)


if __name__ == "__main__":
    main()
