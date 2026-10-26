"""Re-fit the old liquid check (pipette_coach/well_change.py) on our own recordings the way it was first tuned
(scripts/eval_well_change.py): its own code run frame by frame over the whole video, as live (before window, after
window, covered test), on the inputs of scripts/well_change_features.py.

Queries (well, tip arrived t0, tip left t1), from the live tip track's stops (>= --min-s on one well):
  dispense   the stop at each SPACE press (on the labelled well if there is one, else the longest) -> should be liquid
             pipette8: well from the fill order; pipette9: from the tip labels (scripts/build_liquid_dataset.py
             --labels tip), presses --skip left out. pipette9 refills wells that already hold liquid.
  hover      a stop with no SPACE press near it -> should be no liquid
  neighbour  the 8 wells around each dispense over the same visit (wells pressed within 6 s left out) -> no liquid
  dry        --dry-recs: recordings where every SPACE is a dip WITHOUT liquid (empty tip, plunger pressed in the
             well): the stop at each press -> no liquid (the hardest "no": tip in the well, nothing comes out)
Settings: tex_ratio x color_de grid (early_tex = tex_ratio - 0.1, early_de = color_de + 3, as now: 0.8/5 -> 0.7/8).
K-fold: each recording cut into --folds time blocks; a block is scored with the setting that has the best F1 on all
other blocks (both recordings). Last, the best setting on everything (for live).

    python scripts/tune_well_change_live.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.liquid_net import press_well  # noqa: E402
from scripts.eval_well_change import run_causal, stops  # noqa: E402

TEX = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
DE = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def press_labels(rec: str, skip: set) -> dict:
    """press k -> (col, row) labelled well."""
    if rec == "pipette8":
        return {k: press_well(k) for k in range(96)}
    meta = json.loads((PROCESSED_DIR / "liquid_ds" / rec / "meta.json").read_text())
    return {lab["k"]: (lab["col"], lab["row"]) for lab in meta["tip_labels"] if lab["k"] not in skip}


def queries(rec: str, d, labels: dict, press_t: list[float], min_s: float, dry: bool = False) -> list[tuple]:
    """[(well, t0, t1, skip, kind, press k or -1, tip stop on the labelled well)]"""
    sts = stops(d["H"], d["t"], d["tip"][:, None, :], 0.5, min_s)
    out = []
    if dry:  # every press is a dip without liquid: the longest stop at it
        for k, tp in enumerate(press_t):
            near = [s for s in sts if s["t1"] >= tp - 1.5 and s["t0"] <= tp + 0.5]
            if near:
                s = max(near, key=lambda s: s["t1"] - s["t0"])
                out.append((s["w"], s["t0"], s["t1"], [], "dry", k, True))
        return out
    for k, tp in enumerate(press_t):
        near = [s for s in sts if s["t1"] >= tp - 1.5 and s["t0"] <= tp + 0.5]
        if k not in labels or not near:
            continue
        on = [s for s in near if s["w"] == labels[k]]
        s = max(on or near, key=lambda s: s["t1"] - s["t0"])
        busy = [labels[j] for j, tj in enumerate(press_t) if j in labels and abs(tj - tp) <= 6 and j != k]
        out.append((s["w"], s["t0"], s["t1"], busy, "dispense", k, bool(on)))
        c, r = s["w"]
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                w = (c + dc, r + dr)
                if (dc or dr) and 0 <= w[0] < 12 and 0 <= w[1] < 8 and w not in busy:
                    out.append((w, s["t0"], s["t1"], busy + [s["w"]], "neighbour", k, True))
    for s in sts:
        if not any(s["t0"] - 1.5 <= tp <= s["t1"] + 1.5 for tp in press_t):
            out.append((s["w"], s["t0"], s["t1"], [], "hover", -1, True))
    return out


def score(res: list, qs: list, idx) -> dict:
    """liquid found at dispenses; false liquid at hovers / neighbours (None = not seen clearly = not counted)."""
    out = {}
    for kind in ("dispense", "hover", "neighbour", "dry"):
        sel = [i for i in idx if qs[i][4] == kind]
        yes = sum(1 for i in sel if res[i] and res[i]["filled"] is True)
        out[kind] = f"{yes}/{len(sel)}"
    return out


def f1(res, qs, idx) -> float:
    tp = sum(1 for i in idx if qs[i][4] == "dispense" and res[i] and res[i]["filled"] is True)
    fp = sum(1 for i in idx if qs[i][4] != "dispense" and res[i] and res[i]["filled"] is True)
    fn = sum(1 for i in idx if qs[i][4] == "dispense") - tp
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recs", nargs="+", default=["pipette8", "pipette9"])
    ap.add_argument("--dry-recs", nargs="*", default=[], help="recordings where every SPACE is a dip without liquid")
    ap.add_argument("--skip", nargs="*", type=int, default=[48, 60], help="pipette9 presses with doubtful labels")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--min-s", type=float, default=0.4)
    a = ap.parse_args()
    data = {}
    for rec in a.recs + a.dry_recs:
        d = dict(np.load(PROCESSED_DIR / "well_change" / f"{rec}.npz"))
        press_t = [
            m["t"] for m in json.loads((REPO_ROOT / "data" / "own" / rec / "dispenses.json").read_text())["dispenses"]
        ]
        dry = rec in a.dry_recs
        qs = queries(rec, d, {} if dry else press_labels(rec, set(a.skip)), press_t, a.min_s, dry)
        tq = np.array([q[1] for q in qs])
        edges = np.linspace(0, d["t"][-1] + 1e-3, a.folds + 1)
        block = np.searchsorted(edges, tq, side="right") - 1
        data[rec] = {"d": d, "qs": qs, "block": block}
        n = {k: sum(q[4] == k for q in qs) for k in ("dispense", "hover", "neighbour", "dry")}
        agree = sum(q[6] for q in qs if q[4] == "dispense")
        print(f"{rec}: {len(press_t)} presses -> {n}, tip stop on the labelled well at {agree}/{n['dispense']}",
              flush=True)  # fmt: skip
    grid = [(t, de) for t in TEX for de in DE]
    if (0.8, 5.0) not in grid:
        grid.append((0.8, 5.0))
    results = {}
    for g, (t, de) in enumerate(grid):
        kw = {"tex_ratio": t, "color_de": de, "early_tex": round(t - 0.1, 3), "early_de": de + 3}
        results[(t, de)] = {rec: run_causal(v["d"]["F"], v["d"]["t"], [q[:4] for q in v["qs"]], kw)
                            for rec, v in data.items()}  # fmt: skip
        print(f"setting {g + 1}/{len(grid)} tex {t} de {de}:",
              {rec: score(results[(t, de)][rec], v["qs"], range(len(v["qs"]))) for rec, v in data.items()},
              flush=True)  # fmt: skip

    def f1_where(setting, keep) -> float:
        tp = fp = fn = 0
        for rec, v in data.items():
            res, qs = results[setting][rec], v["qs"]
            for i in range(len(qs)):
                if not keep(rec, v["block"][i]):
                    continue
                hit = bool(res[i] and res[i]["filled"] is True)
                if qs[i][4] == "dispense":
                    tp, fn = tp + hit, fn + (not hit)
                else:
                    fp += hit
        return 2 * tp / max(2 * tp + fp + fn, 1)

    folds, pooled_idx = [], {rec: {} for rec in data}
    for rec in data:
        for b in range(a.folds):

            def others(r, bb, rec=rec, b=b) -> bool:
                return not (r == rec and bb == b)

            best = max(grid, key=lambda s, others=others: f1_where(s, others))
            idx = [i for i in range(len(data[rec]["qs"])) if data[rec]["block"][i] == b]
            folds.append(
                {"rec": rec, "block": b, "setting": best, "score": score(results[best][rec], data[rec]["qs"], idx)}
            )
            for i in idx:
                pooled_idx[rec][i] = best
            print(folds[-1], flush=True)
    pooled = {}
    for rec, v in data.items():
        mixed = [results[pooled_idx[rec][i]][rec][i] if i in pooled_idx[rec] else None for i in range(len(v["qs"]))]
        pooled[rec] = {"kfold": score(mixed, v["qs"], range(len(v["qs"]))),
                       "current_0.8_5": score(results[(0.8, 5.0)][rec], v["qs"], range(len(v["qs"])))}  # fmt: skip
    final = max(grid, key=lambda s: f1_where(s, lambda r, b: True))
    print("POOLED:", json.dumps(pooled, indent=1), "\nbest on everything:", final, flush=True)
    run = REPO_ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M%S}_tune_well_change_live"
    run.mkdir(parents=True)
    table = {f"{t}_{de}": {rec: score(results[(t, de)][rec], v["qs"], range(len(v["qs"]))) for rec, v in data.items()}
             for t, de in grid}  # fmt: skip
    (run / "metrics.json").write_text(json.dumps({"args": vars(a), "folds": folds, "pooled": pooled,
                                                  "final": {"tex_ratio": final[0], "color_de": final[1]},
                                                  "all_settings": table}, indent=1, default=str))  # fmt: skip
    print("run:", run)


if __name__ == "__main__":
    main()
