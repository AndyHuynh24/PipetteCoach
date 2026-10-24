"""Click the 4 corners of the plate on frames of a video (trusted labels for the plate segmentation model).

One frame every --every seconds. Click the plate's 4 outer corners in any order (they are sorted around the
centre); click near an existing corner to move it. Nothing is saved until Enter.
  click        place / move a corner (4 max)            Enter / space   save this frame and go to the next
  u            remove the last corner                   x               no plate visible (saved as empty)
  s            skip this frame (not saved)              b               previous frame        q / Esc  quit
Labels: data/annotations/plate_corners_<video>.json {frame index: {"corners": [[x, y] x4] | [], "t": seconds}}.
Frames before --train-until seconds are for training, later ones are held out for testing.

    python scripts/label_plate_corners.py --video data/raw/pipette1.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT  # noqa: E402

VIEW_W = 1280  # window width; frames are scaled to it (labels are stored in original pixels)
LOUPE, LOUPE_ZOOM = 180, 4


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points around their centre (consistent winding)."""
    c = pts.mean(0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


class CornerTool:
    def __init__(self, frames: list[tuple[int, np.ndarray]], fps: float, out: Path, train_until: float):
        self.frames, self.fps, self.out, self.train_until = frames, fps, out, train_until
        self.lab = json.loads(out.read_text()) if out.exists() else {
            "_why": "Hand-clicked plate corners (scripts/label_plate_corners.py); original px. Keep local.",
            "labels": {}}  # fmt: skip
        h, w = frames[0][1].shape[:2]
        self.s = VIEW_W / w
        self.k = 0
        done = set(self.lab["labels"])
        while self.k < len(frames) - 1 and str(frames[self.k][0]) in done:
            self.k += 1
        self.load()
        self.mouse = None
        self.msg = ""

    def load(self) -> None:
        e = self.lab["labels"].get(str(self.frames[self.k][0]))
        self.pts = [np.array(p, float) for p in e["corners"]] if e else []

    def click(self, x: int, y: int) -> None:
        p = np.array([x, y], float) / self.s
        if self.pts:
            d = [np.linalg.norm(p - q) for q in self.pts]
            j = int(np.argmin(d))
            if d[j] < 25 / self.s or len(self.pts) == 4:  # move the nearest corner
                self.pts[j] = p
                return
        self.pts.append(p)

    def save(self) -> bool:
        if len(self.pts) not in (0, 4):
            self.msg = f"Click all 4 corners ({len(self.pts)}/4), or press x for no plate"
            return False
        idx, _ = self.frames[self.k]
        corners = order_corners(np.array(self.pts)).round(2).tolist() if self.pts else []
        self.lab["labels"][str(idx)] = {"corners": corners, "t": round(idx / self.fps, 2)}
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.lab, indent=1))
        return True

    def key(self, k: int) -> bool:
        self.msg = ""
        if k in (13, 10, 32):
            if self.save():
                self.k = min(self.k + 1, len(self.frames) - 1)
                self.load()
        elif k == ord("x"):
            self.pts = []
            self.save()
            self.k = min(self.k + 1, len(self.frames) - 1)
            self.load()
        elif k == ord("u") and self.pts:
            self.pts.pop()
        elif k == ord("s"):
            self.k = min(self.k + 1, len(self.frames) - 1)
            self.load()
        elif k == ord("b"):
            self.k = max(self.k - 1, 0)
            self.load()
        elif k in (ord("q"), 27):
            return False
        return True

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse = (x, y)
            self.click(x, y)

    def render(self) -> np.ndarray:
        idx, f = self.frames[self.k]
        img = cv2.resize(f, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
        for p in self.pts:
            c = tuple(int(v) for v in p * self.s)
            cv2.circle(img, c, 7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(img, c, (0, 0, 255), cv2.MARKER_CROSS, 24, 1)
        if len(self.pts) == 4:
            q = (order_corners(np.array(self.pts)) * self.s).astype(np.int32)
            cv2.polylines(img, [q], True, (0, 255, 0), 2, cv2.LINE_AA)
        if self.mouse is not None:
            mx, my = self.mouse
            r = LOUPE // (2 * LOUPE_ZOOM)
            src = img[max(my - r, 0) : my + r, max(mx - r, 0) : mx + r].copy()
            if src.size:
                lo = cv2.resize(src, (LOUPE, LOUPE), interpolation=cv2.INTER_NEAREST)
                cv2.drawMarker(lo, (LOUPE // 2, LOUPE // 2), (0, 255, 0), cv2.MARKER_CROSS, 18, 1)
                lx = mx + 20 if mx + 20 + LOUPE < img.shape[1] else mx - 20 - LOUPE
                ly = int(np.clip(my - LOUPE // 2, 0, img.shape[0] - LOUPE))
                img[ly : ly + LOUPE, lx : lx + LOUPE] = lo
        t = idx / self.fps
        part = "TRAIN" if t < self.train_until else "TEST (held out)"
        saved = self.lab["labels"].get(str(idx))
        state = "saved" if saved else "not saved"
        n = sum(1 for v in self.lab["labels"].values() if v["corners"])
        head = (
            f"frame {self.k + 1}/{len(self.frames)}  t={t:.0f}s  [{part}]  [{state}]  corners {len(self.pts)}/4  "
            f"labelled {n}   click 4 plate corners, Enter = save & next, u undo, x no plate, s skip, b back, q quit"
        )
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, head, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if self.msg:
            cv2.putText(img, self.msg, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        return img


def load_frames(video: str, every_s: float) -> tuple[list[tuple[int, np.ndarray]], float]:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for idx in range(0, n, max(1, int(every_s * fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            out.append((idx, f))
    return out, fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=float, default=10.0, help="seconds between labelled frames")
    ap.add_argument("--train-until", type=float, default=300.0)
    ap.add_argument("--snapshot", default=None, help="render the first frame to this image and exit")
    a = ap.parse_args()
    frames, fps = load_frames(a.video, a.every)
    out = REPO_ROOT / "data" / "annotations" / f"plate_corners_{Path(a.video).stem}.json"
    tool = CornerTool(frames, fps, out, a.train_until)
    if a.snapshot:
        cv2.imwrite(a.snapshot, tool.render())
        return
    cv2.namedWindow("plate corners", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("plate corners", tool.on_mouse)
    while True:
        cv2.imshow("plate corners", tool.render())
        k = cv2.waitKey(30)
        if k != -1 and not tool.key(k & 0xFF):
            break
    cv2.destroyAllWindows()
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
