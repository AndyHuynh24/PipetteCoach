"""Fine-tune the one-well liquid model (pipette_coach/liquid_net.py LiquidCropNet, ImageNet ResNet18): is liquid going
into THIS well right now? Live, the well comes from the tip model + grid; here it comes from the fill order.

Data: scripts/build_liquid_dataset.py (warped plates, grid corrected, well of each press from the fill order).
  yes  the press's well, every frame within SPACE +- 1 s
  no   in the same frames: a neighbouring well (the tip passes over them), a well filled by an earlier press (liquid
       standing still), any other well
Split by press (one plate, so a time split): --train-to / --val-to; test presses are never loaded here.
Val: every val frame's well + all its neighbours + 4 earlier-filled + 4 other wells.
Output: models/<name>/best.pt and runs/<timestamp>_<name>/ (metrics.json, notes.md written by hand).

    python scripts/train_liquid_crop.py --rec pipette8
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
from train_liquid_net import pairs  # noqa: E402

from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.liquid_net import CROP, LiquidCropNet, S, crop_well, press_well, shift_hue, to_input  # noqa: E402

ALL = [(c, r) for r in range(8) for c in range(12)]


def neighbours(c: int, r: int) -> list[tuple[int, int]]:
    return [(c + dc, r + dr) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dc or dr) and 0 <= c + dc < 12 and 0 <= r + dr < 8]  # fmt: skip


def filled_before(k: int) -> list[tuple[int, int]]:
    """Wells that hold liquid already when press k happens (pressed earlier in the fill order)."""
    return [press_well(j) for j in range(k)]


def train_wells(s: dict, rng: random.Random) -> list[tuple[tuple[int, int], float]]:
    """1 yes + 3 no for one frame."""
    me = (s["col"], s["row"])
    filled = filled_before(s["k"])
    out = [(me, 1.0), (rng.choice(neighbours(*me)), 0.0), (rng.choice([w for w in ALL if w != me]), 0.0)]
    out.append((rng.choice(filled), 0.0) if filled else (rng.choice([w for w in ALL if w != me]), 0.0))
    return out


def val_wells(s: dict, rng: random.Random) -> list[tuple[tuple[int, int], float, str]]:
    me = (s["col"], s["row"])
    nb = neighbours(*me)
    filled = [w for w in filled_before(s["k"]) if w not in nb]
    rest = [w for w in ALL if w != me and w not in nb and w not in filled]
    return ([(me, 1.0, "yes")] + [(w, 0.0, "neighbour") for w in nb]
            + [(w, 0.0, "filled") for w in rng.sample(filled, min(4, len(filled)))]
            + [(w, 0.0, "other") for w in rng.sample(rest, min(4, len(rest)))])  # fmt: skip


def crop_pair(P, s, j, well, rng=None):
    now, before = crop_well(np.array(P[s["idx"]]), *well), crop_well(np.array(P[j]), *well)
    if rng is not None:
        for ax in (0, 1):
            if rng.random() < 0.5:
                now, before = np.flip(now, ax), np.flip(before, ax)
        if rng.random() < 0.7:
            dh = rng.randrange(180)
            now, before = shift_hue(np.ascontiguousarray(now), dh), shift_hue(np.ascontiguousarray(before), dh)
        a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
        T = np.float32([[1, 0, rng.randint(-5, 5)], [0, 1, rng.randint(-5, 5)]])  # grid error
        n = CROP * S
        now, before = (cv2.warpAffine(cv2.convertScaleAbs(np.ascontiguousarray(im), alpha=a, beta=b), T, (n, n),
                                      borderMode=cv2.BORDER_REFLECT) for im in (now, before))  # fmt: skip
    return to_input(np.ascontiguousarray(now), np.ascontiguousarray(before))


@torch.no_grad()
def evaluate(model, P, items, dev, bs=128) -> dict:
    model.eval()
    probs = []
    for b in range(0, len(items), bs):
        X = np.stack([crop_pair(P, s, j, w) for s, j, w, _, _ in items[b : b + bs]])
        probs.append(torch.sigmoid(model(torch.from_numpy(X).to(dev))).cpu().numpy())
    p = np.concatenate(probs)
    y = np.array([t for *_, t, _ in items])
    kind = np.array([k for *_, k in items])
    ks = np.array([s["k"] for s, *_ in items])
    order = np.argsort(-p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(len(p))
    pos, neg = ranks[y == 1], ranks[y == 0]
    auc = float(((len(p) - pos[:, None]) > (len(p) - neg[None, :])).mean())  # P(yes scores above a no)
    out = {"crops": len(p), "auc": round(auc, 4)}
    for thr in (0.5, 0.8):
        out[f"thr{thr}"] = {"yes_found": round(float((p[y == 1] > thr).mean()), 4),
                            **{f"false_{k}": round(float((p[kind == k] > thr).mean()), 4)
                               for k in ("neighbour", "filled", "other")},
                            "presses_seen": f"{len({k for k in ks[(y == 1) & (p > thr)]})}/{len(set(ks))}"}  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="pipette8")
    ap.add_argument("--train-to", type=int, default=64)
    ap.add_argument("--val-to", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--name", default="liquid_crop")
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
    vrng = random.Random(a.seed + 1)
    val = [(s, j, w, t, kind) for s, j in items if a.train_to <= s["k"] < a.val_to
           for w, t, kind in val_wells(s, vrng)]  # fmt: skip
    print(f"train {len(train)} frames x 4 crops / {len({s['k'] for s, _ in train})} presses, val {len(val)} crops",
          flush=True)  # fmt: skip

    model = LiquidCropNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(train) * 4 + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)
    lossf = torch.nn.BCEWithLogitsLoss()
    out_dir = REPO_ROOT / "models" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    run = REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_{a.name}"
    run.mkdir(parents=True)
    hist, best = [], None
    for ep in range(1, a.epochs + 1):
        model.train()
        crops = [(s, j, w, t) for s, j in train for w, t in train_wells(s, rng)]
        rng.shuffle(crops)
        t0, tot = time.time(), 0.0
        for b in range(0, len(crops), a.bs):
            chunk = crops[b : b + a.bs]
            X = torch.from_numpy(np.stack([crop_pair(P, s, j, w, rng) for s, j, w, _ in chunk])).to(dev)
            Y = torch.tensor([t for *_, t in chunk], dtype=torch.float32, device=dev)
            loss = lossf(model(X), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(chunk)
        m = evaluate(model, P, val, dev)
        m.update(epoch=ep, train_loss=round(tot / len(crops), 4), sec=round(time.time() - t0))
        hist.append(m)
        print(m, flush=True)
        if best is None or m["auc"] > best["auc"]:
            best = m
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val": m, "args": vars(a)}, out_dir / "best.pt")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev)["state_dict"])
    model.eval()
    speed = {}
    for d in (dev, torch.device("cpu")):
        mm, x = model.to(d), torch.zeros(1, 6, CROP * S, CROP * S, device=d)
        with torch.no_grad():
            for _ in range(5):
                mm(x).cpu()
            t0 = time.perf_counter()
            for _ in range(60):
                mm(x).cpu()
        speed[f"ms_per_well_{d.type}"] = round((time.perf_counter() - t0) / 60 * 1000, 2)
    (run / "metrics.json").write_text(json.dumps({"rec": a.rec, "split": {"train_presses": [0, a.train_to - 1],
        "val_presses": [a.train_to, a.val_to - 1], "test_presses": f">= {a.val_to} (not run)"}, "args": vars(a),
        "best_val": best, "speed": speed, "history": hist}, indent=1))  # fmt: skip
    print("best:", best, "\nspeed:", speed, "\nrun:", run)


if __name__ == "__main__":
    main()
