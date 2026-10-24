"""Model C: which-well classifier trained only from clip-level dispense labels (multiple-instance learning).

Per frame, a U-Net on the plate view gives a logit map; a well's frame score = max of the map over the well
opening. A well's clip score = soft-max (log-mean-exp, sharpness R) over the clip's frames. Loss: BCE of the
clip scores against labels.json (dispensed wells = 1, all others = 0). So a dispensed well must light up in SOME
frame and every other well in NONE. Trains on all train plates (single + multichannel). Flips are valid
augmentation (the plate view is symmetric about the well grid) with the well labels permuted to match.

Val (plate 4, single wells): prediction = argmax clip score. Metrics: exact accuracy, neighbour rate.
Caveat: C may learn any cue that marks the dispensed well in the clip (tip, or liquid appearing), not only tips.

    python scripts/train_tip_c.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.localizer.detector import _default_device  # noqa: E402
from pipette_coach.localizer.well_net import build_model, to_tensor  # noqa: E402
from pipette_coach.protocol.wells import index_well, well_str  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tracking import plate_view as pv  # noqa: E402
from pipette_coach.tracking.tip_data import clip_wells, load_clip, split_clips  # noqa: E402

R = 5.0  # log-mean-exp sharpness over frames (-> max as R grows)
T_FRAMES = 24
CELL_R = 0.45  # well opening radius, pitch

FLIP_COLS = np.array([r * 12 + (11 - c) for r in range(8) for c in range(12)])
FLIP_ROWS = np.array([(7 - r) * 12 + c for r in range(8) for c in range(12)])


def cell_index():
    """(96, K) flat pixel indices of each well opening in the plate view (padded by repetition)."""
    yy, xx = np.mgrid[0 : pv.H, 0 : pv.W]
    cells = [np.flatnonzero((xx - x) ** 2 + (yy - y) ** 2 <= (CELL_R * pv.PX) ** 2) for x, y in pv.NODES]
    K = max(len(c) for c in cells)
    return np.stack([np.resize(c, K) for c in cells])


def well_scores(model, x, idx):
    """(T, 96) frame scores: max logit over each well opening."""
    L = model(x)[:, 0].flatten(1)  # (T, H*W)
    return L[:, idx.flatten()].view(L.shape[0], 96, -1).amax(-1)


def pool(s, r: float = R):
    import torch

    return (torch.logsumexp(r * s, 0) - np.log(s.shape[0])) / r


def augment(views: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        views, y = views[:, :, ::-1], y[FLIP_COLS]
    if rng.random() < 0.5:
        views, y = views[:, ::-1], y[FLIP_ROWS]
    v = views.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    v = v + rng.normal(0, rng.uniform(0, 4), v.shape)
    return np.clip(v, 0, 255).astype(np.uint8), y


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    wells = clip_wells()
    train = split_clips("train")
    data = {}
    for c in train:
        d = load_clip(c)
        y = np.zeros(96, np.float32)
        y[wells[c]] = 1
        data[c] = (d["views"][d["has_grid"]], y)
    print(f"train: {len(train)} clips ({sum(len(v) for v, _ in data.values())} frames)", flush=True)
    dev = _default_device()
    idx = torch.from_numpy(cell_index()).to(dev)
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = a.epochs * len(train)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    pos_w = torch.tensor(20.0, device=dev)
    t0 = time.time()
    order = []
    for step in range(steps):
        if not order:
            order = list(rng.permutation(len(train)))
        views, y = data[train[order.pop()]]
        pick = np.sort(rng.choice(len(views), min(T_FRAMES, len(views)), replace=False))
        v, yy = augment(views[pick], y, rng)
        s = pool(well_scores(model, to_tensor(list(v)).to(dev), idx))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s, torch.from_numpy(yy).to(dev), pos_weight=pos_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % (5 * len(train)) == 0:
            print(f"epoch {(step + 1) // len(train)}/{a.epochs} loss {float(loss):.4f} ({time.time() - t0:.0f}s)",
                  flush=True)  # fmt: skip
    out = REPO_ROOT / "models" / "tip_c"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "best.pt")

    # ---- val (plate 4)
    model.eval()
    val = split_clips("val")
    rows, ok, nb, n_frames = [], 0, 0, 0
    t1 = time.perf_counter()
    with torch.no_grad():
        for c in val:
            d = load_clip(c)
            views = d["views"][d["has_grid"]]
            n_frames += len(views)
            fs = torch.cat([well_scores(model, to_tensor(list(views[b : b + 16])).to(dev), idx)
                            for b in range(0, len(views), 16)])  # fmt: skip
            s = pool(fs).cpu().numpy()
            pred, true = int(np.argmax(s)), wells[c][0]
            ok += pred == true
            nb += pred != true and max(abs(pred // 12 - true // 12), abs(pred % 12 - true % 12)) == 1
            rows.append((c, well_str(index_well(true)), well_str(index_well(pred)), pred == true, float(s.max())))
    ms = (time.perf_counter() - t1) * 1000 / max(n_frames, 1)
    run = new_run_dir("tip-C")
    write_metrics(run, {"model": "C which-well MIL", "train_clips": len(train), "val_clips": len(val),
                        "val_exact_accuracy": ok / len(val), "val_neighbour_rate": nb / len(val),
                        "ms_per_frame": ms})  # fmt: skip
    lines = ["# Tip model C (which-well, clip-level labels only)", "", f"Val plate 4: exact {ok}/{len(val)}, "
             f"off-by-one {nb}. {ms:.1f} ms/frame.", "", "| clip | true | predicted | ok | score |",
             "|---|---|---|---|---|"]  # fmt: skip
    lines += [f"| {c} | {t} | {p} | {'yes' if h else 'no'} | {sc:.2f} |" for c, t, p, h, sc in rows]
    lines += ["", "Regenerate: `python scripts/train_tip_c.py` (seed 0) -> models/tip_c/best.pt.",
              "Caveat: trained from clip-level labels only; it may use liquid appearing in the well as well as "
              "the tip."]  # fmt: skip
    (run / "notes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
