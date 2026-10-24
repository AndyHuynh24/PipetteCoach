"""Experiment: dispensed well from ring changes.

The heatmap well net detects each well's ring. A clear tip over a well only PARTLY hides its ring (ring score
drops but stays); a hand or the white shaft hides it COMPLETELY. Per frame and well:
  r = well-net heatmap peak within 0.3 pitch of the well centre (plate view);  base = the well's clean level
  (90th percentile of r over all frames of the plate's clips - wells are uncovered most of the time).
  partial: lo <= r < frac * base   (tip candidate)      hidden: r < lo (hand / shaft)
A partial well counts only if the pipette is confirmed on top: the geometric shaft end lies within 2 pitch.
Pick rule per frame among confirmed partial wells: nearest to the traced tip end ("near") or furthest along
the shaft ("far"). Clip prediction = most-voted well.
(frac, lo, rule) are chosen on TRAIN single-well clips (plates 9, 10), then applied unchanged to VAL plate 4.

    python scripts/exp_ring_change.py
"""

import collections
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import plate_of  # noqa: E402
from pipette_coach.protocol.wells import index_well, well_str  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tracking import plate_view as pv  # noqa: E402
from pipette_coach.tracking.tip_data import TIP_DIR, clip_wells, load_clip, split_clips  # noqa: E402

GATE = 0.3 * pv.PX


def ring_scores(clip: str, net) -> np.ndarray:
    """(T, 96) well-net ring score per frame and well (cached)."""
    f = TIP_DIR / f"ring_scores_{clip}.npy"
    if f.exists():
        return np.load(f)
    d = load_clip(clip)
    yy, xx = np.mgrid[0 : pv.H, 0 : pv.W]
    cells = np.stack([((xx - x) ** 2 + (yy - y) ** 2 <= GATE**2) for x, y in pv.NODES]).reshape(96, -1)
    out = np.full((len(d["views"]), 96), np.nan, np.float32)
    for i in np.flatnonzero(d["has_grid"]):
        hm = net.heatmap(d["views"][i]).reshape(-1)
        out[i] = np.where(cells, hm[None], 0).max(1)
    np.save(f, out)
    return out


def frame_pick(r, base, tip, traced, frac, lo, rule):
    if not (np.isfinite(tip[0]) and np.isfinite(traced[0])):
        return None
    partial = (r >= lo) & (r < frac * base)
    near_pipette = np.linalg.norm(pv.NODES - traced, axis=1) <= 2.0 * pv.PX
    cand = np.flatnonzero(partial & near_pipette)
    if not len(cand):
        return None
    if rule == "near":
        return int(cand[np.argmin(np.linalg.norm(pv.NODES[cand] - traced, axis=1))])
    u = traced - tip
    if np.linalg.norm(u) < 1e-6:
        return int(cand[np.argmin(np.linalg.norm(pv.NODES[cand] - traced, axis=1))])
    return int(cand[np.argmax((pv.NODES[cand] - tip) @ (u / np.linalg.norm(u)))])


def evaluate(clips, R, D, base, wells, frac, lo, rule):
    ok = nb = 0
    rows = []
    for c in clips:
        votes = collections.Counter()
        for i in range(len(R[c])):
            if np.isfinite(R[c][i, 0]):
                k = frame_pick(R[c][i], base[plate_of(c)], D[c]["tip"][i], D[c]["traced"][i], frac, lo, rule)
                if k is not None:
                    votes[k] += 1
        t = wells[c][0]
        p = votes.most_common(1)[0][0] if votes else None
        hit = p == t
        near = p is not None and not hit and max(abs(p // 12 - t // 12), abs(p % 12 - t % 12)) == 1
        ok += hit
        nb += near
        rows.append((c, well_str(index_well(t)), "-" if p is None else well_str(index_well(p)), hit))
    return ok, nb, rows


def main() -> None:
    from pipette_coach.localizer.well_net import WellNet

    net = WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    wells = clip_wells()
    train, val = split_clips("train", {9, 10}), split_clips("val")
    R = {c: ring_scores(c, net) for c in train + val}
    D = {c: load_clip(c) for c in train + val}
    base = {pl: np.nanpercentile(np.concatenate([R[c] for c in R if plate_of(c) == pl]), 90, axis=0)
            for pl in {plate_of(c) for c in R}}  # fmt: skip
    for pl, b in sorted(base.items()):
        print(f"plate {pl}: clean ring score median {np.median(b):.2f} (p10 {np.percentile(b, 10):.2f})")
    grid = list(itertools.product([0.5, 0.6, 0.7, 0.8], [0.05, 0.1, 0.2], ["near", "far"]))
    res = []
    for frac, lo, rule in grid:
        ok, nb, _ = evaluate(train, R, D, base, wells, frac, lo, rule)
        res.append((ok, nb, frac, lo, rule))
    res.sort(key=lambda x: (-x[0], -x[1]))
    print("train (27 clips), best settings:", [(f"{ok}/{len(train)} exact, {nb} off-by-one", fr, lo, ru)
                                             for ok, nb, fr, lo, ru in res[:4]])  # fmt: skip
    _, _, frac, lo, rule = res[0]
    ok, nb, rows = evaluate(val, R, D, base, wells, frac, lo, rule)
    print(f"VAL plate 4 with train-chosen (frac {frac}, lo {lo}, rule {rule}): exact {ok}/{len(val)}, off-by-one {nb}")
    run = new_run_dir("ring-change")
    write_metrics(run, {"method": "ring change (partial vs hidden) + pipette confirm", "frac": frac, "lo": lo,
                        "rule": rule, "train_exact": res[0][0], "train_clips": len(train), "val_exact": ok,
                        "val_off_by_one": nb, "val_clips": len(val)})  # fmt: skip
    lines = ["# Ring-change experiment", "", f"Settings chosen on train: frac {frac}, lo {lo}, rule {rule} "
             f"(train {res[0][0]}/{len(train)} exact). Val plate 4: exact {ok}/{len(val)}, off-by-one {nb}.", "",
             "| clip | true | predicted | ok |", "|---|---|---|---|"]  # fmt: skip
    lines += [f"| {c} | {t} | {p} | {'yes' if h else 'no'} |" for c, t, p, h in rows]
    (run / "notes.md").write_text("\n".join(lines) + "\n")
    print("->", run)


if __name__ == "__main__":
    main()
