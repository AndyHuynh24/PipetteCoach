"""Re-fit the liquid check's thresholds (pipette_coach/well_change.py: texture ratio and colour shift, relative to
the plate) on our own recordings, k-fold.

Data: the before/after pairs of scripts/train_well_compare.py (press-clip views of scripts/build_well_views.py,
labels from the fill order). Per pair the rule's two numbers: t_rel (texture after / before, over the plate's median)
and de (colour shift minus the plate's). Liquid = t_rel < tex_ratio or de > color_de.
Fold f picks the thresholds with the best F1 on the other folds' pairs and is scored on its own block; pooled
numbers = all blocks with their own fold's thresholds. Last, thresholds fitted on everything (for live).

    python scripts/tune_well_change.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_compare_kfold import fold_pairs  # noqa: E402
from train_well_compare import liquid_of, make_pairs, report  # noqa: E402

from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.liquid_net import S  # noqa: E402
from pipette_coach.well_change import cell_features  # noqa: E402

TEX = np.round(np.arange(0.5, 1.001, 0.025), 3)
DE = np.round(np.arange(1.0, 15.01, 0.5), 2)


def rule_numbers(views: np.ndarray, pairs: list[tuple]) -> np.ndarray:
    """(n, 2): t_rel, de of each pair (the rule's two numbers, relative to the plate over the same two views)."""
    F = np.stack([cell_features(v, S) for v in views])
    out, memo = np.zeros((len(pairs), 2), np.float32), {}
    for n, (c, r, i1, i2, *_) in enumerate(pairs):
        if (i1, i2) not in memo:
            tex = F[i2, ..., 0] / np.maximum(F[i1, ..., 0], 1e-3)
            sh = F[i2, ..., 2:4] - F[i1, ..., 2:4]
            memo[(i1, i2)] = (tex, sh, np.median(tex), np.median(sh.reshape(-1, 2), 0))
        tex, sh, mt, ms = memo[(i1, i2)]
        out[n] = tex[r, c] / mt, np.linalg.norm(sh[r, c] - ms)
    return out


def verdict(x: np.ndarray, tex_ratio: float, color_de: float) -> np.ndarray:
    return ((x[:, 0] < tex_ratio) | (x[:, 1] > color_de)).astype(np.float32)


def best_thresholds(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """(tex_ratio, color_de, F1) with the best F1 for 'liquid added' over the grid."""
    best = (0.8, 5.0, -1.0)
    for t in TEX:
        for d in DE:
            v = verdict(x, t, d)
            tp, fp, fn = (v * y).sum(), (v * (1 - y)).sum(), ((1 - v) * y).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best[2]:
                best = (float(t), float(d), float(f1))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="pipette8")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--gap", type=int, default=4)
    a = ap.parse_args()
    d = PROCESSED_DIR / "well_views" / a.rec
    clips = json.loads((d / "meta.json").read_text())["clips"]
    views, wdev = np.load(d / "plates.npy"), np.load(d / "dev.npy")
    n, size = 96, 96 // a.folds
    liquid = liquid_of(views, clips)
    folds, all_val, v_new, v_old = [], [], [], []
    for f in range(a.folds):
        lo, hi = f * size, (f + 1) * size if f < a.folds - 1 else n
        train, val = fold_pairs(clips, wdev, lo, hi, n, a.gap)
        xt, yt = rule_numbers(views, train), np.array([p[4] for p in train])
        t, de, f1 = best_thresholds(xt, yt)
        xv = rule_numbers(views, val)
        new, old = verdict(xv, t, de), verdict(xv, 0.8, 5.0)
        m = {"fold": f, "presses": [lo, hi - 1], "tex_ratio": t, "color_de": de, "train_f1": round(f1, 3),
             "refit": report(new, val, liquid), "current": report(old, val, liquid)}  # fmt: skip
        print(m, flush=True)
        folds.append(m)
        all_val += val
        v_new.append(new)
        v_old.append(old)
    pooled = {"refit": report(np.concatenate(v_new), all_val, liquid),
              "current_0.8_5": report(np.concatenate(v_old), all_val, liquid)}  # fmt: skip
    allp = make_pairs(clips, wdev, 0, n, a.gap)
    t, de, f1 = best_thresholds(rule_numbers(views, allp), np.array([p[4] for p in allp]))
    final = {"tex_ratio": t, "color_de": de, "f1_all": round(f1, 3)}
    print("POOLED:", json.dumps(pooled, indent=1), "\nfitted on all presses:", final)
    run = REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_tune_well_change"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"rec": a.rec, "args": vars(a), "folds": folds, "pooled": pooled,
                                                  "final": final}, indent=1))  # fmt: skip
    print("run:", run)


if __name__ == "__main__":
    main()
