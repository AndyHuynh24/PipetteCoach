"""Does the well change (pipette_coach/well_change.py) confirm dispenses on pipette1? Clear water, plus simulated dyes.

Stages (features cache to data/processed/well_change/):
  features  warp each frame of session 1 to the plate cells with the cached live grid (eval_dispenses plate stage)
            -> per-well features. --dye NAME:STRENGTH tints the inside of each reference well from the end of its
            dispense on (a stand-in for coloured liquid; the corners, used for "covered", are never tinted).
  score     1. reference: every reference dispense vs every unfilled well over the same windows (controls)
            2. pipeline: tip stops (eval_dispenses tip cache) -> well change verdict -> dispense log, vs reference

    python scripts/eval_well_change.py features --name water
    python scripts/eval_well_change.py features --name red60 --dye red:0.6
    python scripts/eval_well_change.py score --feats water
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tips import well_of  # noqa: E402
from pipette_coach.well_change import S, WellChange, _masks, cell_features, well_cells  # noqa: E402
from scripts.eval_dispenses import REF, WORK, frames, match, replay, splits, summary, well_name  # noqa: E402

OUT = PROCESSED_DIR / "well_change"
# Light let through per channel (B, G, R) by a strong dye; strength s scales the absorption: x * (1 - s * (1 - T))
DYES = {"red": (0.25, 0.3, 0.95), "blue": (0.95, 0.5, 0.2), "green": (0.4, 0.9, 0.4), "yellow": (0.2, 0.9, 0.95),
        "ink": (0.3, 0.3, 0.3), "milk": None}  # fmt: skip


def rc(name: str) -> tuple[int, int]:
    return "ABCDEFGH".index(name[0]), int(name[1:]) - 1


def tint(cells: np.ndarray, wells: list[tuple[int, int]], dye: str, s: float) -> np.ndarray:
    disc = _masks(S)[0]
    out = cells.astype(np.float32)
    for r, c in wells:
        cell = out[r * S : (r + 1) * S, c * S : (c + 1) * S]
        if DYES[dye] is None:  # milky: scatters light, the well turns pale and flat
            cell[disc] = cell[disc] * (1 - s) + 220 * s
        else:
            cell[disc] = cell[disc] * (1 - s * (1 - np.array(DYES[dye], np.float32)))
    return np.clip(out, 0, 255).astype(np.uint8)


def stage_features(a) -> None:
    pl = np.load(WORK / f"{a.plate}.npz")
    H, t = pl["H"], pl["t"]
    ref = json.loads(Path(a.ref).read_text())["events"]
    dye, strength = (a.dye.split(":")[0], float(a.dye.split(":")[1])) if a.dye else (None, 0.0)
    F = np.full((len(t), 8, 12, 7), np.nan, np.float32)
    for k, (_, tk, f) in enumerate(frames(float(t[0]), float(t[-1]) + 1 / 30, a.ref)):
        if k >= len(t):
            break
        if not np.isfinite(H[k]).all():
            continue
        cells = well_cells(f, H[k])
        if dye:
            cells = tint(cells, [rc(e["well"]) for e in ref if tk >= e["t1"]], dye, strength)
        F[k] = cell_features(cells)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{a.name}.npz", F=F, t=t, plate=a.plate, dye=a.dye or "")
    print(f"saved {OUT / f'{a.name}.npz'}")


def stops(H, t, tips, conf: float, min_s: float, gap_s: float = 0.25) -> list[dict]:
    """Every stop of the most confident tip on one well (gaps <= gap_s bridged) lasting >= min_s."""
    out, cur = [], None
    for k in range(len(t)):
        w = None
        if np.isfinite(H[k, 0, 0]) and np.isfinite(tips[k, 0, 2]) and tips[k, 0, 2] >= conf:
            w = well_of(H[k], tips[k, 0, :2])
        if cur and (w is not None and w != cur["w"] or t[k] - cur["t1"] > gap_s):
            if cur["t1"] - cur["t0"] >= min_s - 1e-6:
                out.append(cur)
            cur = None
        if w is not None:
            if cur is None:
                cur = {"w": w, "t0": float(t[k]), "t1": float(t[k])}
            cur["t1"] = float(t[k])
    if cur and cur["t1"] - cur["t0"] >= min_s - 1e-6:
        out.append(cur)
    return out


def run_causal(F, t, queries, wc_kw: dict) -> list[dict]:
    """Feed features frame by frame, as live; each query (well, t0, t1[, skip]) is decided with WellChange.decide
    from the moment the tip has left (t1) - early when the well is seen clearly. after_s = decision time - t1."""
    wc, res = WellChange(**wc_kw), [None] * len(queries)
    todo, active, j = sorted(range(len(queries)), key=lambda i: queries[i][2]), [], 0
    for k in range(len(t)):
        tk = float(t[k])
        if np.isfinite(F[k, 0, 0, 0]):
            wc.update_features(tk, F[k])
        while j < len(todo) and tk >= queries[todo[j]][2]:
            active.append(todo[j])
            j += 1
        for i in list(active):
            w, t0, t1 = queries[i][:3]
            r = wc.decide(w, t0, t1, tk, skip=queries[i][3] if len(queries[i]) > 3 else ())
            if r is not None:
                res[i] = {**r, "after_s": round(tk - t1, 2)}
                active.remove(i)
    return res


def stage_score(a) -> dict:
    d = np.load(OUT / f"{a.feats}.npz")
    F, t = d["F"], d["t"]
    plate = str(d["plate"])
    refj = json.loads(Path(a.ref).read_text())
    ref = refj["events"]
    lo, hi = refj["covers"]
    wc_kw = json.loads(a.wc) if a.wc else {}
    filled_t = {rc(e["well"]): e["t1"] for e in ref}

    # 1. reference dispenses vs unfilled wells over the same windows
    q = []
    for e in ref:
        R, C = rc(e["well"])
        busy = [(c, r) for (r, c), t1 in filled_t.items() if e["t0"] - 6 <= t1 <= e["t1"] + 4]
        for r in range(8):
            for c in range(12):
                if filled_t.get((r, c), 1e9) < e["t0"] - 0.1 or ((r, c) != (R, C) and (c, r) in busy):
                    continue
                q.append(((c, r), e["t0"], e["t1"], busy, (r, c) == (R, C), e["well"]))
    res = run_causal(F, t, q, wc_kw)
    pos = [(qq[5], rr) for qq, rr in zip(q, res, strict=False) if qq[4]]
    neg = [rr for qq, rr in zip(q, res, strict=False) if not qq[4]]
    pos_f = [p for p in pos if p[1] and p[1]["filled"] is not None]
    neg_f = [n for n in neg if n and n["filled"] is not None]
    refm = {"dispenses": len(pos), "seen": len(pos_f), "confirmed": sum(p[1]["filled"] for p in pos_f),
            "controls": len(neg), "controls_seen": len(neg_f), "controls_filled": sum(n["filled"] for n in neg_f),
            "after_s_dispense": [p[1]["after_s"] for p in pos_f],
            "after_s_controls_p50_p90": np.percentile([n["after_s"] for n in neg], [50, 90]).round(2).tolist(),
            "tex_dispense": [round(p[1]["tex"], 2) for p in pos_f],
            "de_dispense": [round(p[1]["de"], 1) for p in pos_f],
            "tex_controls_p1_p5": np.percentile([n["tex"] for n in neg_f], [1, 5]).round(2).tolist(),
            "de_controls_p95_p99": np.percentile([n["de"] for n in neg_f], [95, 99]).round(1).tolist()}  # fmt: skip
    missed = [p[0] for p in pos if not (p[1] and p[1]["filled"])]

    # 2. pipeline: tip stops -> well change
    pl = np.load(WORK / f"{plate}.npz")
    tips = np.load(WORK / f"{plate}__{a.tips}.npz")["tips"]
    sts = stops(pl["H"], pl["t"], tips, a.conf, a.min_s)
    sres = run_causal(F, t, [(s["w"], s["t0"], s["t1"]) for s in sts], wc_kw)
    change_ev, done = [], set()
    for s, r in zip(sts, sres, strict=False):
        if r and r["filled"] and s["w"] not in done:
            done.add(s["w"])
            change_ev.append({"t": round(s["t0"] + a.min_s, 2), "well": well_name(s["w"]), "tex": round(r["tex"], 2),
                              "de": round(r["de"], 1)})  # fmt: skip

    # dwell + change as a veto: drop a dwell dispense only when its well was seen clearly and did NOT change
    # (the visit = the tip stop the event fired in or, for a revised event, the next stop on that well)
    def visit(e):
        w = rc(e["well"])[::-1]
        s = [s for s in sts if s["w"] == w and s["t1"] >= e["t"] - 0.3 and s["t0"] <= e["t"] + 3]
        return (w, s[0]["t0"], s[0]["t1"]) if s else (w, e["t"] - 0.4, e["t"])

    def veto(evs):
        res = run_causal(F, t, [visit(e) for e in evs], wc_kw)
        return [e for e, r in zip(evs, res, strict=True) if not (r and r["filled"] is False)]

    modes = {"change": change_ev}
    for name, kw in [("dwell", {}), ("dwell_notrip", {"trip_gap_s": None}),
                     ("dwell_fast", {"min_s": 0.2, "trip_gap_s": None})]:  # fmt: skip
        ev = replay(pl["H"], pl["t"], tips, a.conf, kw)
        modes[name], modes[name + "_veto"] = ev, veto(ev)
    pipe = {}
    for name, evs in modes.items():
        m = match(evs, ref)
        pipe[name] = {"all": summary(m, lo, hi), **{n: summary(m, a0, b0) for n, a0, b0 in splits(refj)}}
        pipe[name]["false_events"] = [e for e in m["false"] if lo <= e["t"] < hi]
        pipe[name]["not_correct"] = [{k: r[k] for k in ("well", "t0", "result")} for r in m["rows"]
                                     if r["result"] != "correct"]  # fmt: skip
    metrics = {"feats": a.feats, "dye": str(d["dye"]), "plate": plate, "tips": a.tips, "wc": wc_kw, "min_s": a.min_s,
               "reference": refm, "reference_missed": missed, "stops_checked": len(sts), "pipeline": pipe}  # fmt: skip
    if not a.quiet:
        print(json.dumps({k: v for k, v in refm.items() if not k.startswith(("tex_d", "de_d"))}), "missed:", missed)
        for n, p in pipe.items():
            print(f"{n:10s}", p["all"], "false:", [(e["well"], e["t"]) for e in p["false_events"]])
    if a.save:
        run = new_run_dir(f"well-change-{a.feats}")
        write_metrics(run, metrics)
        print(run)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    f = sub.add_parser("features")
    f.add_argument("--plate", default="final_plate")
    f.add_argument("--ref", default=str(REF))
    f.add_argument("--name", required=True)
    f.add_argument("--dye", help=f"NAME:STRENGTH, NAME in {sorted(DYES)}")
    s = sub.add_parser("score")
    s.add_argument("--feats", required=True)
    s.add_argument("--ref", default=str(REF))
    s.add_argument("--tips", default="all")
    s.add_argument("--conf", type=float, default=0.3)
    s.add_argument("--min-s", type=float, default=0.2, help="shortest tip stop checked for a well change")
    s.add_argument("--wc", help="WellChange settings as JSON")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--save", action="store_true")
    a = ap.parse_args()
    stage_features(a) if a.stage == "features" else stage_score(a)


if __name__ == "__main__":
    main()
