"""Render the full plate pipeline on a video: detector box -> wells -> length / width -> full plate + 12 x 8 grid.

Every --chunk seconds: the plate detector (visible-plate model, multi-scale search) gives the visible plate box;
the well lattice inside it (wells_first.find_plate, pooled over the chunk) gives spacing and directions;
rebuild_from_visible completes the plate (length from width and width from length, real end chosen from the box)
and the 12 x 8 grid. If a chunk is refused (too little visible to tell) the last good grid is kept while the plate
box has not moved (IoU >= --keep-iou); otherwise nothing is drawn (no guess). Every frame, the well detections
are matched to the grid: green = detected, yellow = reconstructed, magenta = A1; magenta outline = rebuilt plate,
thin green = detector box. Output: runs/<ts>_plate-rebuild/plate_rebuild.mp4 + metrics. Keep local.

    python scripts/render_plate_rebuild.py --video data/raw/pipette1.mp4 --seconds 60 \
        --detector models/plate_det_snapshot.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_plate_det import detect, iou  # noqa: E402
from render_all import encode  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.geometry import GRID, pitch_px, project  # noqa: E402
from pipette_coach.localizer.plate_grid import detect_wells_net  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.localizer.wells_box import pool_points, snap_frame  # noqa: E402
from pipette_coach.localizer.wells_first import (  # noqa: E402
    find_plate,
    plate_outline,
    plausible_outline,
    rebuild_from_visible,
    track_grid,
)
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402


def draw(f: np.ndarray, box, state, wells, title: str) -> np.ndarray:
    img = f.copy()
    lw = max(1, round(f.shape[1] / 900))
    if box is not None:
        cv2.rectangle(img, tuple(np.round(box[:2]).astype(int)), tuple(np.round(box[2:]).astype(int)), (0, 255, 0), lw)
    if state is not None:
        cv2.polylines(img, [np.round(state["outline"]).astype(np.int32)], True, (255, 0, 255), 2 * lw, cv2.LINE_AA)
    if wells is not None:
        matched, pos = wells
        r = max(2, int(round(np.linalg.norm(pos[1] - pos[0]) * 0.18)))
        for k, (x, y) in enumerate(pos):
            c = (int(round(x)), int(round(y)))
            if k == 0:
                cv2.circle(img, c, r + 3, (255, 0, 255), -1, cv2.LINE_AA)
            elif matched[k]:
                cv2.circle(img, c, r, (0, 255, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(img, c, r + 1, (0, 255, 255), lw, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34 * lw), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 24 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * lw, (255, 255, 255), lw, cv2.LINE_AA)
    return img


class Pipeline:
    """Detector box -> wells -> rebuilt plate + grid, per chunk, with memory of the last good grid."""

    def __init__(self, cfg: dict, det: PlateDetector, net, keep_iou: float, hold_s: float = 2.0,
                 verify_every: int = 4, replace_margin: int = 5):  # fmt: skip
        self.cfg, self.det, self.net, self.keep_iou, self.hold_s = cfg, det, net, keep_iou, hold_s
        self.verify_every, self.replace_margin = verify_every, replace_margin  # chunks; wells
        self.n_chunks = 0
        self.prev_box = None  # detector box of the previous chunk (plate-moving test)
        self.moving_min_wells = 40
        self.gate = cfg["plate_edges"]["well_gate_pitch"]
        self.last = None  # (state, box) of the last good plate
        self.last_t = -1e9  # time the last good plate was confirmed (rebuilt or wells on it)
        self.stats = {"frames": 0, "rebuilt": 0, "tracked": 0, "memory": 0, "no_grid": 0, "no_box": 0}

    def reset(self) -> None:
        self.last, self.last_t, self.prev_box = None, -1e9, None

    def wells_on_grid(self, f: np.ndarray, state: dict) -> int:
        """Number of grid nodes with a detected well in this frame (well detector at the plate's scale)."""
        s = state["scale"]
        fs = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        dets = detect_wells_net(fs, project(state["H_grid"], GRID[[0, 11, 95, 84]]) * s, self.cfg, self.net) / s
        return int(snap_frame(state["H_grid"], dets, self.gate, 10**9)[1].sum())

    def dets_near(self, frames: list, state: dict) -> np.ndarray:
        """Wells detected around a grid, pooled over a few frames of the chunk (a hand moving over the plate hides
        different wells in different frames)."""
        s, H = state["scale"], state["H_grid"]
        pts = []
        for f in frames[:: max(1, len(frames) // 4)]:
            fs = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            pts.append(detect_wells_net(fs, project(H, GRID[[0, 11, 95, 84]]) * s, self.cfg, self.net) / s)
        return pool_points(pts, pitch_px(H), min_frames=1)

    @staticmethod
    def full(state) -> bool:
        return state is not None and sorted(state.get("counts", [])) == [8, 12]

    def fresh(self, frames: list, box) -> dict | None:
        r = find_plate(frames[:: max(1, len(frames) // 4)], self.det, self.net, self.cfg)
        out = None if r is None else rebuild_from_visible(r, box, self.cfg)
        return None if out is None else {**out, "scale": r["scale"]}

    def track(self, frames: list, state: dict) -> dict | None:
        tr = track_grid(state["H_grid"], self.dets_near(frames, state), self.cfg)
        if tr is None:
            return None
        return {"H_grid": tr[0], "outline": plate_outline(tr[0], self.cfg), "scale": state["scale"], "n": tr[1]}

    def run_clip(self, chunks: list[list], fps: float, title: str) -> list[np.ndarray]:
        """Recorded clip (all frames available, like the approved method that pooled a whole clip): anchor at the
        chunk where the plate is rebuilt most completely (all 12 x 8 visible, most wells on the grid), then track
        forward and backward from it with the identified wells (memory decides which side each visible well is
        on, so stray dots such as multichannel tips cannot turn the grid). A fresh rebuild replaces the tracked
        grid only when it shows the full plate and fits clearly more wells (the plate was really moved)."""
        n = len(chunks)
        mids = [c[len(c) // 2] for c in chunks]
        boxes = [(lambda b: b[0] if len(b) else None)(detect(self.det, m)) for m in mids]
        fresh = [self.fresh(c, b) if b is not None else None for c, b in zip(chunks, boxes, strict=True)]
        score = [-1 if f is None else self.wells_on_grid(m, f) + (1000 if self.full(f) else 0)
                 for f, m in zip(fresh, mids, strict=True)]  # fmt: skip
        states: list = [None] * n
        srcs = ["no plate box" if b is None else "plate box, but too little visible to rebuild" for b in boxes]
        if max(score) >= 0:
            a = int(np.argmax(score))
            states[a], srcs[a] = fresh[a], f"anchor: rebuilt from visible wells {fresh[a]['counts']}"
            for order in (range(a + 1, n), range(a - 1, -1, -1)):
                prev, last_k = fresh[a], a
                for k in order:
                    tr = self.track(chunks[k], prev)
                    f = fresh[k]
                    if (
                        f is not None
                        and self.full(f)
                        and (
                            tr is None
                            or self.wells_on_grid(mids[k], f) >= self.wells_on_grid(mids[k], tr) + self.replace_margin
                        )
                    ):
                        states[k], srcs[k] = f, f"rebuilt from visible wells {f['counts']}"  # fmt: skip
                    elif tr is not None:
                        states[k], srcs[k] = tr, f"tracked: {tr['n']} visible wells identified on the anchored grid"
                    elif abs(k - last_k) * len(chunks[k]) / fps <= self.hold_s:
                        states[k], srcs[k] = prev, "held (plate hidden; grid from the neighbouring moment)"
                    else:
                        continue
                    if not srcs[k].startswith("held"):
                        prev, last_k = states[k], k
        imgs = []
        for k in range(n):
            src = srcs[k]
            if src.startswith(("rebuilt", "anchor")):
                key = "rebuilt"
            elif src.startswith("tracked"):
                key = "tracked"
            elif src.startswith("held"):
                key = "memory"
            else:
                key = "no_box" if boxes[k] is None else "no_grid"
            imgs += self.render_chunk(chunks[k], boxes[k], states[k], src, key, k * len(chunks[0]) / fps, fps,
                                      title)  # fmt: skip
        return imgs

    def chunk(self, frames: list, t0: float, fps: float, title: str = "") -> list[np.ndarray]:
        cfg, det, net = self.cfg, self.det, self.net
        mid = frames[len(frames) // 2]
        boxes = detect(det, mid)
        box = boxes[0] if len(boxes) else None
        state, src = None, "no plate box"
        self.n_chunks += 1
        # 1. memory: identify the visible wells on the remembered grid and refit the grid to them
        tracked = None
        if self.last is not None:
            tr = track_grid(self.last[0]["H_grid"], self.dets_near(frames, self.last[0]), cfg)
            if tr is not None:
                tracked = {"H_grid": tr[0], "outline": plate_outline(tr[0], cfg), "scale": self.last[0]["scale"],
                           "n": tr[1]}  # fmt: skip
        # 2. fresh rebuild: no memory, tracking lost (plate moved / lifted), or a periodic check that replaces the
        #    tracked grid only if it fits clearly more wells
        new = None
        if box is not None and (tracked is None or self.n_chunks % self.verify_every == 0):
            r = find_plate(frames[:: max(1, len(frames) // 4)], det, net, cfg)
            out = None if r is None else rebuild_from_visible(r, box, cfg)
            new = None if out is None else {**out, "scale": r["scale"]}
            if new is not None and tracked is not None:
                if self.wells_on_grid(mid, new) < self.wells_on_grid(mid, tracked) + self.replace_margin:
                    new = None
        still = self.last is not None and box is not None and iou(box, self.last[1]) >= self.keep_iou
        # plate moving (lifted / carried: the box jumps between chunks) -> blurry, tilted wells: only trust a grid
        # backed by many wells that looks like a plate, or a fresh rebuild that sees the whole 12 x 8; otherwise
        # draw nothing (a wrong grid is worse than none)
        moving = box is not None and self.prev_box is not None and iou(box, self.prev_box) < self.keep_iou
        self.prev_box = box
        if moving:
            if tracked is not None and not (tracked["n"] >= self.moving_min_wells
                                            and plausible_outline(tracked["outline"], box)):  # fmt: skip
                tracked = None
            if new is not None and not self.full(new):
                new = None
            still = False
        if new is not None:
            state, src = new, f"rebuilt from visible wells {new['counts']}"
        elif tracked is not None:
            state, src = tracked, f"tracked: {tracked['n']} visible wells identified on the remembered grid"
        elif still:
            state, src = self.last[0], "kept (plate did not move; too few wells visible)"
        elif self.last is not None and not moving and t0 - self.last_t <= self.hold_s:
            state, src = self.last[0], f"held (plate hidden, last seen {t0 - self.last_t:.1f} s ago)"
        elif box is not None:
            src = "plate box, but too little visible to rebuild"
        if src.startswith(("rebuilt", "tracked")):
            self.last, self.last_t = (state, box if box is not None else self.last[1]), t0
        if src.startswith("rebuilt"):
            key = "rebuilt"
        elif src.startswith("tracked"):
            key = "tracked"
        elif src.startswith(("kept", "held")):
            key = "memory"
        else:
            key = "no_box" if box is None else "no_grid"
        return self.render_chunk(frames, box, state, src, key, t0, fps, title)

    def render_chunk(self, frames, box, state, src, key, t0, fps, title) -> list[np.ndarray]:
        cfg, net = self.cfg, self.net
        out_imgs = []
        for i, f in enumerate(frames):
            wells = None
            if state is not None:
                s = state["scale"]
                fs = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                quad = project(state["H_grid"], GRID[[0, 11, 95, 84]]) * s
                dets = detect_wells_net(fs, quad, cfg, net) / s
                _, matched, pos = snap_frame(state["H_grid"], dets, self.gate, 10**9)
                wells = (matched, pos)
            n_det = f"{int(wells[0].sum())}/96 wells detected, " if wells is not None else ""
            out_imgs.append(draw(f, box, state, wells, f"{title}t={t0 + i / fps:5.1f}s  {n_det}{src}"))
            self.stats[key] += 1
        self.stats["frames"] += len(frames)
        return out_imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="a video file (e.g. data/raw/pipette1.mp4)")
    ap.add_argument("--plates", default=None, help="OR dataset plates, e.g. 4,1,2,9,10 (train/val clips only)")
    ap.add_argument("--seconds", type=float, default=60, help="video mode; 0 = whole video")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--chunk", type=float, default=0.5)
    ap.add_argument("--keep-iou", type=float, default=0.8)
    ap.add_argument("--detector", default="models/plate_det/weights/best.pt")
    ap.add_argument("--name", default="plate-rebuild")
    ap.add_argument("--clips", default="", help="dataset mode: only these clips (comma-separated)")
    a = ap.parse_args()
    cfg = load_config("localizer")
    cfg["detector"]["weights"] = a.detector
    pipe = Pipeline(cfg, PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt"), a.keep_iou)
    run = new_run_dir(a.name)
    vw = None
    t_start = time.time()

    def write(imgs: list[np.ndarray], fps: float) -> None:
        nonlocal vw
        for img in imgs:
            if vw is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(str(run / "raw.mp4"), fourcc, fps, (img.shape[1], img.shape[0]))
            vw.write(img)

    per_clip = {}
    if a.plates:
        from pipette_coach.data import clips_for_split, load_splits, plate_of
        from pipette_coach.localizer.session import read_frames

        splits = load_splits()
        plates = [int(x) for x in a.plates.split(",") if x]
        if any(p in splits["test"] for p in plates):
            sys.exit("a test plate was requested; refusing (test plates stay untouched).")
        fps = 30.0
        for pl in plates:
            only = {c for c in a.clips.split(",") if c}
            for clip in [c for sp in ("train", "val") for c in clips_for_split(sp) if plate_of(c) == pl]:
                if only and clip not in only:
                    continue
                frames = read_frames(clip)
                pipe.reset()  # every clip starts without memory
                before = dict(pipe.stats)
                n_chunk = max(1, int(a.chunk * fps))
                chunks = [frames[k : k + n_chunk] for k in range(0, len(frames), n_chunk)]
                imgs = pipe.run_clip(chunks, fps, f"{clip}  ")
                write(imgs, fps)
                d = {k: pipe.stats[k] - before[k] for k in pipe.stats}
                per_clip[clip] = d
                print(f"{clip}: {d} ({time.time() - t_start:.0f}s)", flush=True)
        src = f"plates {a.plates}"
    else:
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
        n_total = int(a.seconds * fps) if a.seconds > 0 else n_frames - int(a.start * fps)
        n_chunk, done = max(1, int(a.chunk * fps)), 0
        while done < n_total:
            frames = []
            for _ in range(min(n_chunk, n_total - done)):
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            if not frames:
                break
            write(pipe.chunk(frames, a.start + done / fps, fps), fps)
            done += len(frames)
            if done % (n_chunk * 60) < n_chunk:
                print(f"{done}/{n_total} frames {pipe.stats} ({time.time() - t_start:.0f}s)", flush=True)
        src = a.video
    vw.release()
    encode(run / "raw.mp4", run / "plate_rebuild.mp4")
    st = pipe.stats
    share = (st["rebuilt"] + st["tracked"] + st["memory"]) / max(st["frames"], 1)
    m = {"source": src, "detector": a.detector, **st, "grid_share": share, "per_clip": per_clip}
    write_metrics(run, m)
    summary = {k: v for k, v in m.items() if k != "per_clip"}
    (run / "notes.md").write_text(f"# Plate rebuild render ({src})\n\n{summary}. Contains dataset / third-party "
                                  "frames: keep local.\n")  # fmt: skip
    print("->", run / "plate_rebuild.mp4", summary)


if __name__ == "__main__":
    main()
