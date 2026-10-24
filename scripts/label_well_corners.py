"""Check / fix the 96-well grid on a long video, one frame every --every seconds (labels for retraining the well net).

Each frame opens with a proposed grid and its 4 corner wells A1, A12, H12, H1 as handles:
  orange = the live plate tracker's grid on that frame (run once over the video, cached),
  cyan   = your grid from the frame before, when you fixed it there and the plate box has not moved since.
If the grid sits on the wells, press Enter. If not, drag a corner handle onto the centre of that corner well's TOP
OPENING (the rim circle, not the dark floor spot); clicking anywhere moves the nearest handle there. The grid is
refit from the 4 corners and nudged onto the wells' opening rims (green).
  drag/click   move the nearest corner                  Enter / space   save this frame and go to the next
  u            back to the proposal                     x               can't tell / plate moving / no plate
  t            switch tracker / your previous grid      c               clear, click A1, A12, H12, H1 from scratch
  z            zoom 3x around the cursor / back         r               opening snap on/off
  f            turn the labelling 180 deg (A1 <-> H12), when the printed letters say A1 is the other corner
  b            previous frame                           q / Esc         quit
  n            another frame of the same clip (clip mode, scripts/label_clip_grids.py; the grid is kept)
Labels (saved on every Enter / x): data/annotations/well_corners_<video>.json
{frame index: {"t": s, "H": 3x3 canonical (col, row) -> px | null, "corners": [[x, y] x4] | null,
"source": tracker | carried | adjusted | clicked | x}} (source says how often the tracker grid was already right).
Tracker proposals: data/processed/well_label_proposals/<video>_<every>s_<rows>.npz (regenerated if missing).

    python scripts/label_well_corners.py --video data/raw/pipette4.mp4 --rows-toward up
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR  # noqa: E402
from pipette_coach.localizer.geometry import GRID, grid_error, pitch_px, project  # noqa: E402
from pipette_coach.localizer.openings import refine_to_openings  # noqa: E402

CORNERS = np.array([[0, 0], [11, 0], [11, 7], [0, 7]], np.float32)  # A1, A12, H12, H1 in canonical (col, row)
NAMES = ["A1", "A12", "H12", "H1"]
COLOURS = {"tracker": (0, 165, 255), "carried": (255, 255, 0), "adjusted": (0, 255, 0), "clicked": (0, 255, 0)}
VIEW_W = 1280  # window width; frames are scaled to it (labels are stored in original pixels)
LOUPE, LOUPE_ZOOM = 200, 4
BOX_STILL = 0.3  # plate box moved less than this (in well spacings) -> your previous grid is proposed


def stretches(labels: dict, tol: float = 0.1) -> list[dict]:
    """Saved frames -> stretches {"f0", "f1", "H"}: consecutive saved frames whose grids stay within tol well
    spacing of the stretch's first grid form one stretch; every frame from f0 to f1 gets that first grid.
    An x (H null) ends the stretch."""
    out: list[dict | None] = []
    for idx in sorted(labels, key=int):
        H = labels[idx]["H"]
        if H is None:
            out.append(None)
            continue
        last = out[-1] if out else None
        if last is not None and grid_error(np.array(last["H"]), np.array(H), ignore_labelling=False) <= tol:
            last["f1"] = int(idx)
        else:
            out.append({"f0": int(idx), "f1": int(idx), "H": H})
    return [s for s in out if s is not None]


class WellCornerTool:
    def __init__(self, indices: list[int], read: Callable[[int], np.ndarray], fps: float, out: Path, cfg: dict,
                 size: tuple[int, int], proposals: dict | None = None):  # fmt: skip
        """proposals: {frame index: (H | None, plate box x0 y0 x1 y1 | None)} from the tracker."""
        self.indices, self.read, self.fps, self.out, self.cfg = indices, read, fps, out, cfg
        self.proposals = proposals or {}
        self.lab = json.loads(out.read_text()) if out.exists() else {
            "_why": "Hand-checked 96-well grids from the 4 corner wells A1, A12, H12, H1 "
                    "(scripts/label_well_corners.py); original px. Keep local.",
            "labels": {}}  # fmt: skip
        self.size = size
        self.view = (0.0, 0.0, float(size[0]))  # x0, y0, width of the shown part (original px); kept across frames
        self.s = VIEW_W / size[0]
        self.snap, self.mouse, self.drag, self.msg, self.n_snap = True, None, None, "", 0
        self.n_alt, self.alt = 0, 0  # clip mode: read(idx + alt) shows other frames of the same item (n key)
        self.k = 0
        while self.k < len(indices) - 1 and str(indices[self.k]) in self.lab["labels"]:
            self.k += 1
        self.load()

    # ---- proposals ----
    def options(self) -> dict[str, np.ndarray]:
        """Proposed grids for this frame, preferred first."""
        idx = self.indices[self.k]
        H_tr, box = self.proposals.get(idx, (None, None))
        opts = {}
        if self.k > 0:
            prev = self.lab["labels"].get(str(self.indices[self.k - 1]))
            if prev and prev["H"] is not None and prev["source"] in ("carried", "adjusted", "clicked"):
                H_prev = np.array(prev["H"])
                _, box_prev = self.proposals.get(self.indices[self.k - 1], (None, None))
                still = (box is not None and box_prev is not None
                         and np.abs(np.subtract(box, box_prev)).max() < BOX_STILL * pitch_px(H_prev))  # fmt: skip
                if still or H_tr is None:
                    opts["carried"] = H_prev
        if H_tr is not None:
            opts["tracker"] = np.array(H_tr)
        if self.k > 0 and not opts:  # nothing else: your grid from the frame before, if any
            prev = self.lab["labels"].get(str(self.indices[self.k - 1]))
            if prev and prev["H"] is not None:
                opts["carried"] = np.array(prev["H"])
        return opts

    def propose(self, source: str | None = None) -> None:
        opts = self.options()
        source = source if source in opts else next(iter(opts), None)
        self.source = source or "clicked"
        self.H = opts[source] if source else None
        self.corners = [] if self.H is None else list(project(self.H, CORNERS).astype(float))

    def load(self) -> None:
        self.alt = 0
        self.frame = self.read(self.indices[self.k])
        e = self.lab["labels"].get(str(self.indices[self.k]))
        if e and e["H"] is not None:
            self.H, self.source = np.array(e["H"]), e["source"]
            self.corners = [np.array(p, float) for p in e["corners"]]
        else:
            self.propose()

    # ---- editing ----
    def _fit(self, snap: bool) -> np.ndarray | None:
        if len(self.corners) < 4:
            return None
        H, _ = cv2.findHomography(CORNERS, np.array(self.corners, np.float32), 0)
        self.n_snap = 0
        if snap and self.snap:
            H, self.n_snap = refine_to_openings(self.frame, H, self.cfg)
        return H

    def to_orig(self, x: float, y: float) -> np.ndarray:
        return np.array([x, y], float) / self.s + self.view[:2]

    def to_view(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, float) - self.view[:2]) * self.s

    def zoom(self, x: int, y: int) -> None:
        """3x around the view point (x, y), or back to the whole frame."""
        W, Hh = self.size
        if self.view[2] < W:
            self.view = (0.0, 0.0, float(W))
        else:
            cw, c = W / 3, self.to_orig(x, y)
            ch = cw * Hh / W
            self.view = (float(np.clip(c[0] - cw / 2, 0, W - cw)), float(np.clip(c[1] - ch / 2, 0, Hh - ch)), cw)
        self.s = VIEW_W / self.view[2]

    def press(self, x: int, y: int) -> None:
        p = self.to_orig(x, y)
        if len(self.corners) < 4:  # clicking from scratch: A1, A12, H12, H1 in order
            self.corners.append(p)
            self.source = "clicked"
            self.H = self._fit(snap=True)
            return
        self.drag = int(np.argmin([np.linalg.norm(p - q) for q in self.corners]))
        self.move(x, y)

    def move(self, x: int, y: int) -> None:
        if self.drag is not None:
            self.corners[self.drag] = self.to_orig(x, y)
            self.H = self._fit(snap=False)  # quick while dragging

    def release(self) -> None:
        if self.drag is not None:
            self.drag = None
            if self.source != "clicked":
                self.source = "adjusted"
            self.H = self._fit(snap=True)

    def _save(self, H: np.ndarray | None) -> None:
        idx = self.indices[self.k]
        self.lab["labels"][str(idx)] = {
            "t": round(idx / self.fps, 2),
            "H": None if H is None else np.round(H, 8).tolist(),
            "corners": None if H is None else np.round(self.corners, 2).tolist(),
            "source": "x" if H is None else self.source,
        }
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.lab, indent=1))

    def _go(self, step: int) -> None:
        self.k = int(np.clip(self.k + step, 0, len(self.indices) - 1))
        self.load()

    def key(self, k: int) -> bool:
        self.msg = ""
        if k in (13, 10, 32):
            if self.H is None:
                self.msg = f"Click all 4 corner wells ({len(self.corners)}/4), or press x"
            else:
                self._save(self.H)
                self._go(1)
        elif k == ord("x"):
            self._save(None)
            self._go(1)
        elif k == ord("u"):
            if self.source == "clicked" and self.corners:
                self.corners.pop()
                self.H = self._fit(snap=True)
            else:
                self.propose()
        elif k == ord("t"):
            opts = list(self.options())
            if self.source in opts and len(opts) > 1:
                self.propose(opts[(opts.index(self.source) + 1) % len(opts)])
            elif opts:
                self.propose(opts[0])
        elif k == ord("f") and len(self.corners) == 4:
            self.corners = self.corners[2:] + self.corners[:2]
            self.source = "adjusted" if self.source != "clicked" else "clicked"
            self.H = self._fit(snap=True)
        elif k == ord("c"):
            self.corners, self.H, self.source = [], None, "clicked"
        elif k == ord("r"):
            self.snap = not self.snap
            if self.source in ("adjusted", "clicked"):
                self.H = self._fit(snap=True)
        elif k == ord("z"):
            self.zoom(*(self.mouse or (VIEW_W // 2, int(VIEW_W * self.size[1] / self.size[0]) // 2)))
        elif k == ord("n") and self.n_alt:
            self.alt = (self.alt + 1) % self.n_alt
            self.frame = self.read(self.indices[self.k] + self.alt)
        elif k == ord("b"):
            self._go(-1)
        elif k in (ord("q"), 27):
            return False
        return True

    def on_mouse(self, event, x, y, flags, param) -> None:
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.press(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.release()

    # ---- drawing ----
    def render(self) -> np.ndarray:
        x0, y0, cw = (int(round(v)) for v in self.view)
        vh = int(round(VIEW_W * self.size[1] / self.size[0]))
        crop = self.frame[y0 : y0 + int(round(cw * self.size[1] / self.size[0])), x0 : x0 + cw]
        img = cv2.resize(crop, (VIEW_W, vh), interpolation=cv2.INTER_AREA)
        if self.H is not None:
            col = COLOURS[self.source]
            r = max(3, int(0.2 * pitch_px(self.H) * self.s))
            for x, y in self.to_view(project(self.H, GRID)):
                cv2.circle(img, (int(x), int(y)), r, col, 2, cv2.LINE_AA)
        for name, p in zip(NAMES, self.corners, strict=False):  # corner handles
            c = tuple(int(v) for v in self.to_view(p))
            cv2.circle(img, c, 9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, name, (c[0] + 10, c[1] - 10), 0, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        if self.mouse is not None:  # magnifier next to the cursor, from the full-res frame
            mx, my = self.mouse
            fx, fy = (int(v) for v in self.to_orig(mx, my))
            rr = LOUPE // (2 * LOUPE_ZOOM)
            src = self.frame[max(fy - rr, 0) : fy + rr, max(fx - rr, 0) : fx + rr]
            if src.size:
                lo = cv2.resize(src, (LOUPE, LOUPE), interpolation=cv2.INTER_CUBIC)
                cv2.drawMarker(lo, (LOUPE // 2, LOUPE // 2), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
                lx = mx + 20 if mx + 20 + LOUPE < img.shape[1] else mx - 20 - LOUPE
                ly = int(np.clip(my - LOUPE // 2, 0, img.shape[0] - LOUPE))
                img[ly : ly + LOUPE, lx : lx + LOUPE] = lo
        idx = self.indices[self.k]
        saved = self.lab["labels"].get(str(idx))
        state = "not saved" if saved is None else f"saved: {saved['source']}"
        if self.H is None:
            todo = f"click {NAMES[len(self.corners)]}" if len(self.corners) < 4 else ""
        else:
            what = {"tracker": "tracker grid", "carried": "your grid from before"}.get(self.source, "your grid")
            todo = f"{what}: Enter if right, else drag a corner"
        names = self.lab.get("clips")
        where = f"{names[idx // 1000]} view {self.alt + 1}/{self.n_alt}" if names else f"t={idx / self.fps:.1f}s"
        head = (f"frame {self.k + 1}/{len(self.indices)}  {where}  [{state}]  {todo}   "
                "u undo, t switch, f flip, c clear, x can't tell, z zoom, b back, q quit")  # fmt: skip
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, head, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if self.msg:
            cv2.putText(img, self.msg, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        return img


def video_reader(video: str) -> tuple[Callable[[int], np.ndarray], float, int, tuple[int, int]]:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        return f if ok else np.zeros((size[1], size[0], 3), np.uint8)

    return read, fps, n, size


def tracker_proposals(video: str, indices: list[int], rows_toward: str | None, path: Path) -> dict:
    """Run the live PlateTracker over every frame (as live_demo) and keep its grid + plate box at the given frames.
    Cached at path."""
    if path.exists():
        z = np.load(path)
        if list(z["idx"]) == indices:
            return {int(i): (None if np.isnan(H).any() else H, None if np.isnan(b).any() else b)
                    for i, H, b in zip(z["idx"], z["H"], z["box"], strict=True)}  # fmt: skip
    from pipette_coach.localizer.detector import PlateDetector
    from pipette_coach.localizer.live import PlateTracker
    from pipette_coach.localizer.well_net import WellNet

    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    tr = PlateTracker(cfg, det, net, rows_toward=rows_toward, verify_every=30)
    cap = cv2.VideoCapture(video)
    want, Hs, boxes, t0 = set(indices), {}, {}, time.time()
    for i in range(max(indices) + 1):
        ok, f = cap.read()
        if not ok:
            break
        o = tr.update(f)
        if i in want:
            Hs[i] = o["H"] if o["H"] is not None else np.full((3, 3), np.nan)
            boxes[i] = np.asarray(tr.box, float) if tr.box is not None else np.full(4, np.nan)
        if i % 500 == 0:
            print(f"tracker proposals: frame {i}/{max(indices)}  {time.time() - t0:.0f} s", flush=True)
    nan_H, nan_b = np.full((3, 3), np.nan), np.full(4, np.nan)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, idx=np.array(indices), H=np.array([Hs.get(i, nan_H) for i in indices]),
                        box=np.array([boxes.get(i, nan_b) for i in indices]))  # fmt: skip
    return tracker_proposals(video, indices, rows_toward, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=float, default=2.0, help="seconds between labelled frames")
    ap.add_argument("--rows-toward", choices=["down", "up", "left", "right"], default=None,
                    help="as live_demo: image direction of rows A -> H")  # fmt: skip
    ap.add_argument("--snapshot", default=None, help="render the first unlabelled frame to this image and exit")
    a = ap.parse_args()
    read, fps, n, size = video_reader(a.video)
    indices = list(range(0, n, max(1, int(round(a.every * fps)))))
    stem = Path(a.video).stem
    cache = PROCESSED_DIR / "well_label_proposals" / f"{stem}_{a.every:g}s_{a.rows_toward or 'default'}.npz"
    props = tracker_proposals(a.video, indices, a.rows_toward, cache)
    out = REPO_ROOT / "data" / "annotations" / f"well_corners_{stem}.json"
    tool = WellCornerTool(indices, read, fps, out, load_config("localizer"), size, props)
    if a.snapshot:
        cv2.imwrite(a.snapshot, tool.render())
        return
    cv2.namedWindow("well corners", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("well corners", tool.on_mouse)
    while True:
        cv2.imshow("well corners", tool.render())
        k = cv2.waitKey(30)
        if k != -1 and not tool.key(k & 0xFF):
            break
    cv2.destroyAllWindows()
    src = [v["source"] for v in tool.lab["labels"].values()]
    print(f"saved -> {out}  {len(src)} frames: " + ", ".join(f"{s} {src.count(s)}" for s in sorted(set(src))))


if __name__ == "__main__":
    main()
