"""Train and evaluate the heatmap well-centre net (pipette_coach/localizer/well_net.py).

Train data: frames (every 3rd) of the train clips whose stored grid the ring check marked OK
(data/processed/label_ring_check.json) - no suspect labels. Per frame: wells with visible evidence (clean ring or
ring-filter peak within 0.3 pitch of the grid) are positives (Gaussian, sigma 1.5 px); wells without evidence
(under a hand/pipette) are IGNORED in the loss; everything else, incl. the gaps between wells, is negative.
Augmentation: synthetic fills on visible wells, scale 0.8-1.25, flips/rot90, brightness/contrast, blur, noise.
--video-labels: also hand-checked frames of own videos (scripts/build_well_labels.py: user's grid, covered wells
ignored), resized so a well spacing is pitch_px_nominal as the live tracker does; train split only, --video-share of
each batch. Their val split (by time) is evaluated too. Plates in configs/excluded_plates.json are never used.
--transfyr-labels: the dataset clips from the user's hand-checked grids (data/processed/well_labels/transfyr.json,
covered wells ignored) instead of the ring-confirmed stored grids. --use-splits train val test: final model on all
labels (its val numbers are then on training data). --compare: one or more older models.

Val (plate 4, never trained on): truth = the independent grid-fit reference (plate_autolabels/val.json).
Metrics per method (net, hand-made clean-ring detector, ring filter): precision (detections on wells), share of
detections in gaps, recall of evidence wells, centre error; on real and on SYNTHETIC-filled val frames.

    python scripts/train_well_net.py
    python scripts/train_well_net.py --perspective 0.15 --video-labels pipette1 pipette4 --out well_net_v3 \
        --compare models/well_net/best.pt
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clips_for_split, excluded_plates, plate_of  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector, _default_device  # noqa: E402
from pipette_coach.localizer.geometry import GRID, pitch_px, plate_bbox, project  # noqa: E402
from pipette_coach.localizer.openings import detect_rings  # noqa: E402
from pipette_coach.localizer.rings import find_well_candidates  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.localizer.synthetic import DYES  # noqa: E402
from pipette_coach.localizer.well_net import WellNet, build_model, to_tensor  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402

SIGMA = 1.5
PATCH = 160


def evidence_wells(frame: np.ndarray, H: np.ndarray, cfg: dict, det: PlateDetector) -> np.ndarray:
    """Bool (96,): well has visible evidence (clean ring or ring-filter peak within 0.3 pitch)."""
    p = pitch_px(H)
    g = project(H, GRID)
    rcfg = copy.deepcopy(cfg)
    rcfg["wells"]["method"] = "ring"
    P, _, _ = find_well_candidates(frame, rcfg, plate_bbox(H, 1.0))
    d = det.detect(frame)
    R = detect_rings(frame, d[0], cfg) if d is not None else np.zeros((0, 2))
    E = np.concatenate([a for a in (P, R) if len(a)]) if (len(P) or len(R)) else np.zeros((0, 2))
    if len(E) == 0:
        return np.zeros(len(GRID), bool)
    return np.linalg.norm(g[:, None] - E[None], axis=2).min(1) < 0.3 * p


def plate_crop(frame: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = plate_bbox(H, 1.0)
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    return frame[y0:y1, x0:x1].copy(), np.array([x0, y0], np.float32)


def paint_points(img: np.ndarray, pts: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """SYNTHETIC liquid in the given (visible) wells: dye / dark / milky / half, random per well."""
    out = img.astype(np.float32)
    for x, y in pts:
        if rng.random() < 0.4:
            continue
        k = rng.integers(4)
        dye = DYES[rng.integers(len(DYES))]
        kinds = [(dye, 0.6, 0.33), ((40, 25, 20), 0.9, 0.33), ((235, 235, 235), 0.85, 0.33), (dye, 0.55, 0.24)]
        col, a, r = kinds[k]  # dye / dark / milky / half-filled
        m = np.zeros(img.shape[:2], np.float32)
        cv2.circle(m, (int(round(x * 16)), int(round(y * 16))), int(r * p * 16), 1.0, -1, cv2.LINE_AA, 4)
        m = cv2.GaussianBlur(m, (0, 0), 0.8)[..., None] * a
        out = out * (1 - m) + np.array(col, np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def make_target(shape: tuple[int, int], pos: np.ndarray, ign: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.zeros((h, w), np.float32)
    for x, y in pos:
        t = np.maximum(t, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA**2)))
    mask = np.ones((h, w), np.float32)
    for x, y in ign:
        cv2.circle(mask, (int(round(x)), int(round(y))), int(round(0.45 * p)), 0.0, -1)
    return t, mask


PERSPECTIVE = 0.0  # set by --perspective: max corner shift (share of the crop) of a random perspective warp


def sample(item: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img, pos, ign, p = item["img"], item["pos"].copy(), item["ign"].copy(), item["pitch"]
    if rng.random() < 0.5:
        img = paint_points(img, pos, p, rng)
    s = rng.uniform(0.8, 1.25)  # scale augmentation (other camera distances)
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    pos, ign, p = pos * s, ign * s, p * s
    h, w = img.shape[:2]
    if PERSPECTIVE > 0 and rng.random() < 0.7:  # camera tilted / plate lifted: warp image and wells together
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = (src + rng.uniform(-PERSPECTIVE, PERSPECTIVE, (4, 2)) * min(w, h)).astype(np.float32)
        Hp = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, Hp, (w, h), borderMode=cv2.BORDER_REFLECT)
        if len(pos):
            pos = cv2.perspectiveTransform(pos[None].astype(np.float64), Hp)[0]
        if len(ign):
            ign = cv2.perspectiveTransform(ign[None].astype(np.float64), Hp)[0]
    if h < PATCH or w < PATCH:
        img = cv2.copyMakeBorder(img, 0, max(0, PATCH - h), 0, max(0, PATCH - w), cv2.BORDER_REFLECT)
        h, w = img.shape[:2]
    y0, x0 = rng.integers(0, h - PATCH + 1), rng.integers(0, w - PATCH + 1)
    img = img[y0 : y0 + PATCH, x0 : x0 + PATCH]
    pos, ign = pos - [x0, y0], ign - [x0, y0]
    t, m = make_target((PATCH, PATCH), pos, ign, p)
    t, m = t.astype(np.float32), m.astype(np.float32)  # MPS has no float64
    k = rng.integers(4)  # rot90 / flips (the well lattice is symmetric)
    img, t, m = np.rot90(img, k), np.rot90(t, k), np.rot90(m, k)
    if rng.random() < 0.5:
        img, t, m = img[:, ::-1], t[:, ::-1], m[:, ::-1]
    img = img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    img = img + rng.normal(0, rng.uniform(0, 4), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), np.ascontiguousarray(t), np.ascontiguousarray(m)


def evaluate(method, frames_truth: list, cfg: dict, det: PlateDetector, fill: bool, rng: np.random.Generator) -> dict:
    prec, gap, rec, err = [], [], [], []
    for f, H, ev in frames_truth:
        p = pitch_px(H)
        g = project(H, GRID)
        gaps = project(H, (GRID[:-13] + GRID[13:]) / 2)
        img = paint_points(f, g[ev], p, rng) if fill else f
        D = method(img, H)
        if len(D) == 0:
            rec.append(0.0)
            continue
        dw = np.linalg.norm(D[:, None] - g[None], axis=2)
        on = dw.min(1) < 0.25 * p
        prec.append(float(on.mean()))
        gap.append(float((np.linalg.norm(D[:, None] - gaps[None], axis=2).min(1) < 0.25 * p).mean()))
        found = np.linalg.norm(g[ev][:, None] - D[None], axis=2).min(1) < 0.25 * p if ev.any() else np.zeros(0, bool)
        rec.append(float(found.mean()) if ev.any() else 1.0)
        err += list(dw.min(1)[on])
    return {"precision": float(np.mean(prec)), "in_gaps": float(np.mean(gap)), "recall_visible": float(np.mean(rec)),
            "centre_err_px": float(np.median(err)) if err else None}  # fmt: skip


def video_frames(video: str, split: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(frame resized to the nominal well spacing, H in that frame, used wells) for the hand-checked frames of an own
    video in the given time split (data/processed/well_labels/<video>.json); frames with no clean reference skipped."""
    lab = json.loads((PROCESSED_DIR / "well_labels" / f"{video}.json").read_text())
    nominal = load_config("localizer")["wells"]["pitch_px_nominal"]
    cap = cv2.VideoCapture(str(REPO_ROOT / "data" / "raw" / f"{video}.mp4"))
    out = []
    for e in lab["frames"]:
        if e["split"] != split or e["used"] is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["idx"])
        ok, f = cap.read()
        if not ok:
            continue
        H = np.array(e["H"])
        s = nominal / pitch_px(H)
        f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.append((f, np.diag([s, s, 1.0]) @ H, np.array(e["used"], bool)))
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perspective", type=float, default=0.0, help="max corner shift of perspective warps (0.15)")
    ap.add_argument("--out", default="well_net", help="model folder in models/ (keep the old one: well_net_v2)")
    ap.add_argument("--compare", nargs="*", default=["models/well_net/best.pt"], help="older models to compare with")
    ap.add_argument("--transfyr-labels", action="store_true", help="dataset clips from the hand-checked grids")
    ap.add_argument("--use-splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    ap.add_argument("--video-labels", nargs="*", default=[], help="own videos with hand-checked grids")
    ap.add_argument("--video-share", type=float, default=0.5, help="share of each batch from the own videos")
    ap.add_argument("--eval-only", action="store_true", help="no training: evaluate models/<out>/best.pt")
    ap.add_argument("--no-eval", action="store_true", help="train and save only (e.g. the final all-label model)")
    ap.add_argument(
        "--nets-only",
        action="store_true",
        help="eval: the nets on real frames only (no synthetic fills,"
        " no hand-made ring detectors - those take ~30 min)",
    )
    a = ap.parse_args()
    global PERSPECTIVE
    PERSPECTIVE = a.perspective
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    cfg = load_config("localizer")
    det = PlateDetector(cfg)
    grids = json.loads((PROCESSED_DIR / "clip_grids.json").read_text())
    ok = [r["clip"] for r in json.loads((PROCESSED_DIR / "label_ring_check.json").read_text()) if r["verdict"] == "OK"]
    ok = [c for c in ok if plate_of(c) not in set(excluded_plates())]
    items = []
    t0 = time.time()
    if a.eval_only:
        ok, a.video_labels = [], []
    elif a.transfyr_labels:  # hand-checked grids, covered wells ignored
        tl = [e for e in json.loads((PROCESSED_DIR / "well_labels" / "transfyr.json").read_text())["frames"]
              if e["split"] in a.use_splits]  # fmt: skip
        ok = sorted({e["clip"] for e in tl})
        for clip in ok:
            frames = read_frames(clip, 3)
            for e in (e for e in tl if e["clip"] == clip):
                H, used = np.array(e["H"]), np.array(e["used"], bool)
                crop, off = plate_crop(frames[e["frame"] // 3], H)
                g = project(H, GRID) - off
                items.append({"img": crop, "pos": g[used], "ign": g[~used], "pitch": pitch_px(H)})
    for clip in [] if (a.transfyr_labels or a.eval_only) else ok:
        H = np.array(grids[clip]["H"])
        for f in read_frames(clip, 3):
            ev = evidence_wells(f, H, cfg, det)
            crop, off = plate_crop(f, H)
            g = project(H, GRID) - off
            items.append({"img": crop, "pos": g[ev], "ign": g[~ev], "pitch": pitch_px(H)})
    src = "hand-checked" if a.transfyr_labels else "ring-confirmed"
    print(f"train: {len(ok)} {src} clips ({'/'.join(a.use_splits)}), {len(items)} frames ({time.time() - t0:.0f}s)")
    vitems = []
    for v, split in ((v, sp) for v in a.video_labels for sp in a.use_splits):
        for f, H, used in video_frames(v, split):
            crop, off = plate_crop(f, H)
            g = project(H, GRID) - off
            vitems.append({"img": crop, "pos": g[used], "ign": g[~used], "pitch": pitch_px(H)})
    print(f"train: {len(vitems)} hand-checked own-video frames ({', '.join(a.video_labels) or 'none'})", flush=True)

    dev = _default_device()
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps_per_epoch = max(len(items) // 16, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * steps_per_epoch)
    for ep in range(0 if a.eval_only else a.epochs):
        model.train()
        tot = 0.0
        for _ in range(steps_per_epoch):
            pick = [vitems[rng.integers(len(vitems))] if vitems and rng.random() < a.video_share
                    else items[rng.integers(len(items))] for _ in range(16)]  # fmt: skip
            batch = [sample(it, rng) for it in pick]
            x = to_tensor([b[0] for b in batch]).to(dev)
            t = torch.from_numpy(np.stack([b[1] for b in batch])[:, None]).to(dev)
            m = torch.from_numpy(np.stack([b[2] for b in batch])[:, None]).to(dev)
            pr = torch.sigmoid(model(x))
            w = (1 + 30 * t) * m  # positives are rare: weight them up; ignored wells weigh 0
            loss = (w * (pr - t) ** 2).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / steps_per_epoch:.5f} ({time.time() - t0:.0f}s)", flush=True)
    out = REPO_ROOT / "models" / a.out
    out.mkdir(parents=True, exist_ok=True)
    if not a.eval_only:
        torch.save(model.state_dict(), out / "best.pt")
    if a.no_eval:
        run = new_run_dir(a.out.replace("_", "-"))
        info = {"train_clips": len(ok), "train_frames": len(items), "video_labels": a.video_labels,
                "video_frames": len(vitems), "transfyr_labels": a.transfyr_labels, "use_splits": a.use_splits,
                "epochs": a.epochs, "perspective": a.perspective, "seed": a.seed}  # fmt: skip
        write_metrics(run, info)
        (run / "notes.md").write_text(
            f"# Heatmap well net (trained, not evaluated)\n\n{info}\n\nWeights: {out}/best.pt. "
            "Regenerate: rerun this script with the same arguments.\n"
        )
        print("->", run)
        return
    t_eval = time.time()

    def step(what: str) -> None:
        print(f"eval: {what} ({time.time() - t_eval:.0f}s)", flush=True)

    # ---- val (plate 4): independent grid-fit reference
    val = json.loads((PROCESSED_DIR / "plate_autolabels" / "val.json").read_text())
    vt = []
    for clip in clips_for_split("val"):
        good = [x for x in val.get(clip, []) if x["H"] and x["frac"] >= 0.9]
        if not good:
            continue
        H = np.array(max(good, key=lambda x: x["frac"])["H"])
        for f in read_frames(clip, 8):
            vt.append((f, H, evidence_wells(f, H, cfg, det)))
    step(f"plate 4 reference: {len(vt)} frames")
    net = WellNet(out / "best.pt")
    prevs = [(Path(c).parent.name, WellNet(REPO_ROOT / c)) for c in a.compare if (REPO_ROOT / c).exists()]
    rcfg = copy.deepcopy(cfg)
    rcfg["wells"]["method"] = "ring"

    def m_net(img, H):
        return net.detect(img, plate_bbox(H, 1.0))[0]

    def m_prev(model):
        return lambda img, H: model.detect(img, plate_bbox(H, 1.0))[0]

    # tilted val: every val frame re-filmed by a camera tilted 15-35 deg about the plate centre (truth warped too)
    from eval_generalization import camera_warp

    trng = np.random.default_rng(7)
    vt_tilt = []
    for f, H, ev in vt:
        c = project(H, np.array([[5.5, 3.5]]))[0]
        W = camera_warp(c, 1.0, trng.uniform(-20, 20), trng.uniform(15, 35), trng.uniform(0, 360), c, f=1500.0)
        vt_tilt.append((cv2.warpPerspective(f, W, (f.shape[1], f.shape[0]), borderMode=cv2.BORDER_REPLICATE),
                        W @ H, ev))  # fmt: skip

    def m_rings(img, H):
        d = det.detect(img)
        return detect_rings(img, d[0], cfg) if d is not None else np.zeros((0, 2))

    def m_ringfilter(img, H):
        return find_well_candidates(img, rcfg, plate_bbox(H, 1.0))[0]

    res = {}
    methods = [("heatmap_net", m_net)] + [(f"heatmap_net_previous_{n}", m_prev(m)) for n, m in prevs]
    baselines = [] if a.nets_only else [("clean_rings", m_rings), ("ring_filter", m_ringfilter)]
    for fill in (False,) if a.nets_only else (False, True):
        for name, fn in methods + baselines:
            res[f"{name}{'_synthetic_fill' if fill else ''}"] = evaluate(
                fn, vt, cfg, det, fill, np.random.default_rng(1)
            )
            step(f"{name}{' synthetic fill' if fill else ''}")
    for name, fn in methods:
        res[f"{name}_tilted"] = evaluate(fn, vt_tilt, cfg, det, False, np.random.default_rng(1))
        step(f"{name} tilted")
    tf = PROCESSED_DIR / "well_labels" / "transfyr.json"
    if tf.exists():  # plate 4 with the user's hand-checked grids; truth = visible wells (covered ignored)
        tv = [e for e in json.loads(tf.read_text())["frames"] if e["split"] == "val"][::3]
        cache, vu = {}, []
        for e in tv:
            if e["clip"] not in cache:
                cache = {e["clip"]: read_frames(e["clip"], 3)}
            vu.append((cache[e["clip"]][e["frame"] // 3], np.array(e["H"]), np.array(e["used"], bool)))
        for name, fn in methods:
            res[f"{name}_plate4_handchecked"] = evaluate(fn, vu, cfg, det, False, np.random.default_rng(1))
            step(f"{name} plate 4 hand-checked ({len(vu)} frames)")
    for v in ("pipette1", "pipette4"):  # own videos, val split by time, user's grid; truth = visible wells
        if (PROCESSED_DIR / "well_labels" / f"{v}.json").exists():
            vv = video_frames(v, "val")
            for name, fn in methods:
                res[f"{name}_{v}_val"] = evaluate(fn, vv, cfg, det, False, np.random.default_rng(1))
                step(f"{name} {v} val ({len(vv)} frames)")
    t0 = time.perf_counter()
    for f, H, _ in vt[:20]:
        m_net(f, H)
    ms = (time.perf_counter() - t0) * 1000 / 20
    run = new_run_dir(a.out.replace("_", "-"))
    write_metrics(run, {"train_clips": len(ok), "train_frames": len(items), "val_frames": len(vt), "epochs": a.epochs,
                        "video_labels": a.video_labels, "video_frames": len(vitems), "video_share": a.video_share,
                        "transfyr_labels": a.transfyr_labels, "use_splits": a.use_splits, "compare": a.compare,
                        "perspective": a.perspective, "net_ms_per_plate": ms, **res})  # fmt: skip
    lines = ["# Heatmap well net", "", f"Train: {len(ok)} {src} clips ({'/'.join(a.use_splits)}), {len(items)} "
             f"frames + {len(vitems)} own-video frames. Val: plate 4, "
             f"{len(vt)} frames, independent grid-fit reference. Net: {ms:.0f} ms per plate crop.", "",
             "| method | precision | in gaps | recall (visible wells) | centre error px |",
             "|---|---|---|---|---|"]  # fmt: skip
    for k, v in res.items():
        lines.append(
            f"| {k} | {v['precision']:.1%} | {v['in_gaps']:.1%} | {v['recall_visible']:.1%} | "
            f"{v['centre_err_px'] or 0:.2f} |"
        )
    lines += ["", "Synthetic fills are rendered (same painter as training augmentation): optimistic for the net."]
    (run / "notes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
