"""Click tool for pipette-tip ends (labels for tip model B and the pipette YOLO). Train clips only.

Window: [plate view x3.5, model-B heatmap] [camera crop] + magnifier, timeline and buttons.
Nothing is saved until you press Next:
  click            place the tip marker (click again to move it). Multichannel clips: click the FIRST tip, then
                   the LAST tip (the tips in between are filled in); a third click starts over.
  Save & Next  / Enter      save this frame's label and go to the next queued frame
  No tip       / x          mark "no tip over the plate" (then Save & Next)
  Use suggestion / s        put the marker on the model's suggestion (cyan cross)
  Clear        / c          remove the marker (and the saved label of this frame)
  Play         / p          play the ~1.5 s before this frame, then stop here
  Prev         / b          previous queued frame (its saved label is shown and can be changed)
  a / d                     previous / next frame of this clip      h heatmap   g hint (dispensed well)   q quit
Labels: data/annotations/tip_clicks.json (image px + plate-view px). Resumes where you stopped.

    python scripts/label_tips.py [--plates 9,10,1,2]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_plate_grids import load_grid_windows  # noqa: E402

from pipette_coach.config import REPO_ROOT  # noqa: E402
from pipette_coach.data import load_splits  # noqa: E402
from pipette_coach.localizer.plate_grid import window_at  # noqa: E402
from pipette_coach.localizer.session import read_frames  # noqa: E402
from pipette_coach.protocol.wells import index_well, well_str  # noqa: E402
from pipette_coach.tracking import plate_view as pv  # noqa: E402
from pipette_coach.tracking.tip_data import clip_wells, load_clip, split_clips  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "annotations" / "tip_clicks.json"
ZV = 3.5  # plate-view zoom
VW, VH = int(pv.W * ZV), int(pv.H * ZV)  # 784 x 588
CW = 600  # camera panel width
TOP = 40
TL_H = 24  # timeline
BTN_H = 44
W_TOTAL = VW + CW
H_TOTAL = TOP + VH + 8 + TL_H + 8 + BTN_H + 26
PLAY_FRAMES, PLAY_FPS = 20, 12
LOUPE, LOUPE_ZOOM = 150, 4  # magnifier size (px) and zoom


class Model:
    """Current tip model (clicks-trained if it exists, else the auto-label one) for heatmap + suggestion."""

    def __init__(self, enabled: bool = True):
        self.m = None
        for w in (REPO_ROOT / "models" / "tip_b_clicks" / "best.pt", REPO_ROOT / "models" / "tip_b" / "best.pt"):
            if enabled and w.exists():
                import torch

                from pipette_coach.localizer.detector import _default_device
                from pipette_coach.localizer.well_net import build_model

                self.dev = _default_device()
                self.m = build_model()
                self.m.load_state_dict(torch.load(w, map_location="cpu"))
                self.m.eval().to(self.dev)
                break

    def heatmap(self, view: np.ndarray) -> np.ndarray | None:
        if self.m is None:
            return None
        import torch

        from pipette_coach.localizer.well_net import to_tensor

        with torch.no_grad():
            return torch.sigmoid(self.m(to_tensor([view]).to(self.dev)))[0, 0].cpu().numpy()


def build_queue(clips: list[str], per_clip: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    """Round-robin over clips: per clip, frames spread over the clip where a pipette is in view, plus one without."""
    per = {}
    for c in clips:
        d = load_clip(c)
        has = np.flatnonzero(d["has_grid"])
        tip = np.flatnonzero(d["has_grid"] & np.isfinite(d["tip"][:, 0]))
        notip = np.setdiff1d(has, tip)
        pick = list(tip[np.linspace(0, len(tip) - 1, min(per_clip - 1, len(tip))).astype(int)]) if len(tip) else []
        if len(notip):
            pick.append(int(rng.choice(notip)))
        per[c] = [int(i) for i in pick]
    q = []
    while any(per.values()):
        for c in clips:
            if per[c]:
                q.append((c, per[c].pop(0)))
    return q


BUTTONS = [("prev", "< Prev (b)"), ("play", "Play (p)"), ("none", "No tip (x)"), ("sugg", "Use suggestion (s)"),
           ("clear", "Clear (c)"), ("next", "Save & Next > (Enter)")]  # fmt: skip


class LabelTool:
    def __init__(self, queue, wells, out: Path, model: Model, target: int = 200):
        self.queue, self.wells, self.out, self.model, self.target = queue, wells, out, model, target
        self.lab = json.loads(out.read_text()) if out.exists() else {
            "_why": "Hand-clicked pipette tip ends on TRAIN clips (scripts/label_tips.py). Dataset has no licence: "
                    "positions only, no frames; keep local.", "labels": {}}  # fmt: skip
        self.cache: dict = {}
        self.heat: dict = {}
        self.show_heat, self.show_hint = True, False
        self.mouse = None
        self.play: list[int] = []
        self.play_t = 0.0
        self.msg = ""
        bw = W_TOTAL // len(BUTTONS)
        y0 = TOP + VH + 8 + TL_H + 8
        self.btn_rects = {k: (i * bw + 4, y0, (i + 1) * bw - 4, y0 + BTN_H) for i, (k, _) in enumerate(BUTTONS)}
        q = 0
        done = {(c, int(f)) for c, v in self.lab["labels"].items() for f in v}
        while q < len(queue) - 1 and queue[q] in done:
            q += 1
        self.goto(q)

    # ---------- data
    def clip_data(self, c):
        if c not in self.cache:
            self.cache.clear()
            self.cache[c] = (load_clip(c), read_frames(c), load_grid_windows(c))
        return self.cache[c]

    def goto(self, k: int) -> None:
        self.q = int(np.clip(k, 0, len(self.queue) - 1))
        self.clip, self.frame = self.queue[self.q]
        self.load_pending()

    def load_pending(self) -> None:
        """Pending marker = this frame's saved label (so a revisited frame shows what you saved)."""
        e = self.lab["labels"].get(self.clip, {}).get(str(self.frame))
        self.pending_kind = None if e is None else e["kind"]
        tips = [] if e is None else [np.array(p, float) for p in e["tips_view"]]
        if e is not None and e["kind"] == "tip" and len(tips) > 2:
            tips = [tips[0], tips[-1]]  # multichannel: edit by first + last tip
        self.pending = tips
        self.play = []

    def n_labelled(self) -> int:
        return sum(len(v) for v in self.lab["labels"].values())

    def multi(self) -> int:
        return len(self.wells[self.clip])

    def geometry(self):
        d, frames, wins = self.clip_data(self.clip)
        H = window_at(wins, self.frame)["H_grid"]
        M = pv.view_matrix(H)
        corners = cv2.perspectiveTransform(np.array([[[-1.8, -1.6], [12.8, -1.6], [12.8, 8.6], [-1.8, 8.6]]],
                                                    np.float64), H)[0]  # fmt: skip
        h, w = frames[0].shape[:2]
        x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int), [w, h])
        s = min(CW / (x1 - x0), VH / (y1 - y0))
        return M, (int(x0), int(y0), int(x1), int(y1), s)

    def suggestion(self, frame: int):
        key = (self.clip, frame)
        if key not in self.heat:
            self.heat[key] = self.model.heatmap(self.clip_data(self.clip)[0]["views"][frame])
        hm = self.heat[key]
        if hm is None or hm.max() < 0.15:
            return hm, None
        k = np.unravel_index(np.argmax(hm), hm.shape)
        return hm, np.array([k[1], k[0]], float)

    # ---------- actions
    def tips_to_save(self) -> list[np.ndarray]:
        if self.pending_kind == "tip" and self.multi() > 1 and len(self.pending) == 2:
            a0, a1 = self.pending
            return [a0 + (a1 - a0) * k / (self.multi() - 1) for k in range(self.multi())]
        return self.pending

    def save(self) -> bool:
        if self.pending_kind is None:
            self.msg = "Nothing to save: click the tip, or press No tip / Use suggestion"
            return False
        if self.pending_kind == "tip" and self.multi() > 1 and len(self.pending) < 2:
            self.msg = f"Multichannel: click the FIRST and the LAST tip ({len(self.pending)}/2 clicked)"
            return False
        tips = self.tips_to_save()
        M, _ = self.geometry()
        Minv = np.linalg.inv(M)
        tips_img = [cv2.perspectiveTransform(np.array([[p]], np.float64), Minv)[0, 0].round(2).tolist() for p in tips]
        self.lab["labels"].setdefault(self.clip, {})[str(self.frame)] = {
            "kind": self.pending_kind, "tips_view": [list(map(float, np.round(p, 2))) for p in tips],
            "tips_img": tips_img, "dispensed": [well_str(index_well(k)) for k in self.wells[self.clip]],
        }  # fmt: skip
        self.write()
        return True

    def write(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.lab, indent=1))

    def action(self, name: str) -> bool:
        """Returns False to quit."""
        self.msg = ""
        if name == "next":
            if self.save():
                saved = f"saved {self.clip} frame {self.frame}"
                self.goto(self.q + 1)
                self.msg = saved
        elif name == "prev":
            self.goto(self.q - 1)
        elif name == "none":
            self.pending_kind, self.pending = "none", []
        elif name == "sugg":
            _, s = self.suggestion(self.frame)
            if s is None:
                self.msg = "No suggestion for this frame"
            elif self.multi() > 1:
                self.msg = "Suggestions are single-tip only: click first and last tip"
            else:
                self.pending_kind, self.pending = "tip", [s]
        elif name == "clear":
            self.pending_kind, self.pending = None, []
            if self.lab["labels"].get(self.clip, {}).pop(str(self.frame), None) is not None:
                self.write()
                self.msg = "label removed"
        elif name == "play":
            self.play = list(range(max(0, self.frame - PLAY_FRAMES), self.frame + 1))
            self.play_t = time.time()
        elif name in ("fprev", "fnext"):
            T = len(self.clip_data(self.clip)[1])
            self.frame = int(np.clip(self.frame + (1 if name == "fnext" else -1), 0, T - 1))
            self.load_pending()
        elif name == "heat":
            self.show_heat = not self.show_heat
        elif name == "hint":
            self.show_hint = not self.show_hint
        elif name == "quit":
            return False
        return True

    def click_at(self, x: int, y: int) -> None:
        """Mouse click in window coordinates: a button, or the tip marker in either image panel."""
        for k, (x0, y0, x1, y1) in self.btn_rects.items():
            if x0 <= x < x1 and y0 <= y < y1:
                self.action(k)
                return
        p = self.window_to_view(x, y)
        if p is None:
            return
        self.play = []
        if self.pending_kind != "tip" or self.multi() == 1 or len(self.pending) >= 2:
            self.pending = [p]  # single: (re)place the marker; multi: start over at the first tip
        else:
            self.pending.append(p)
        self.pending_kind = "tip"
        self.msg = ""

    def window_to_view(self, x: int, y: int):
        yy = y - TOP
        if not (0 <= yy < VH and 0 <= x < W_TOTAL):
            return None
        if x < VW:
            return np.array([x / ZV, yy / ZV])
        M, (x0, y0, x1, y1, s) = self.geometry()
        img = np.array([(x - VW) / s + x0, yy / s + y0])
        return cv2.perspectiveTransform(np.array([[img]], np.float64), M)[0, 0]

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse = (x, y)
            self.click_at(x, y)

    def key(self, k: int) -> bool:
        m = {13: "next", 10: "next", 32: "next", ord("b"): "prev", ord("x"): "none", ord("s"): "sugg",
             ord("c"): "clear", ord("p"): "play", ord("a"): "fprev", ord("d"): "fnext", ord("h"): "heat",
             ord("g"): "hint", ord("q"): "quit", 27: "quit"}  # fmt: skip
        return self.action(m[k]) if k in m else True

    # ---------- drawing
    def render(self) -> np.ndarray:
        frame = self.frame
        playing = bool(self.play)
        if playing:
            k = int((time.time() - self.play_t) * PLAY_FPS)
            if k >= len(self.play):
                self.play = []
                playing = False
            else:
                frame = self.play[k]
        d, frames, _ = self.clip_data(self.clip)
        M, (x0, y0, x1, y1, s) = self.geometry()
        V = cv2.resize(d["views"][frame], (VW, VH), interpolation=cv2.INTER_CUBIC)
        hm, sug = (None, None) if playing else self.suggestion(frame)
        if hm is not None and self.show_heat:
            col = cv2.resize(cv2.applyColorMap((np.clip(hm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                             (VW, VH))  # fmt: skip
            al = cv2.resize(np.clip(hm, 0, 1), (VW, VH))[..., None] * 0.5
            V = (V * (1 - al) + col * al).astype(np.uint8)
        crop = cv2.resize(frames[frame][y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        cam = np.zeros((VH, CW, 3), np.uint8)
        cam[: min(VH, crop.shape[0]), : min(CW, crop.shape[1])] = crop[:VH, :CW]
        Minv = np.linalg.inv(M)

        def to_cam(p):
            q = cv2.perspectiveTransform(np.array([[p]], np.float64), Minv)[0, 0]
            return int((q[0] - x0) * s), int((q[1] - y0) * s)

        if self.show_hint and not playing:
            for kk in self.wells[self.clip]:
                cv2.circle(V, tuple((pv.NODES[kk] * ZV).astype(int)), int(0.5 * pv.PX * ZV), (0, 255, 0), 1)
        if sug is not None:
            cv2.drawMarker(V, tuple((sug * ZV).astype(int)), (255, 255, 0), cv2.MARKER_CROSS, 22, 1)
            cv2.drawMarker(cam, to_cam(sug), (255, 255, 0), cv2.MARKER_CROSS, 16, 1)
        if not playing:  # your marker(s): red crosshair + ring, in both panels
            for p in self.tips_to_save() if self.pending_kind == "tip" else []:
                c = (int(p[0] * ZV), int(p[1] * ZV))
                cv2.drawMarker(V, c, (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
                cv2.circle(V, c, 9, (0, 0, 255), 2)
                cv2.drawMarker(cam, to_cam(p), (0, 0, 255), cv2.MARKER_CROSS, 22, 1)
                cv2.circle(cam, to_cam(p), 6, (0, 0, 255), 2)
        canvas = np.full((H_TOTAL, W_TOTAL, 3), 25, np.uint8)
        canvas[TOP : TOP + VH, :VW] = V
        canvas[TOP : TOP + VH, VW:] = cam
        if playing:
            cv2.putText(canvas, f"PLAYING  frame {frame}  (stops at {self.frame})", (12, TOP + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)  # fmt: skip
        if self.pending_kind == "none" and not playing:
            cv2.putText(canvas, "NO TIP", (VW // 2 - 60, TOP + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # magnifier around the mouse (over either image panel)
        if self.mouse is not None and not playing and TOP <= self.mouse[1] < TOP + VH:
            mx, my = self.mouse
            r = LOUPE // (2 * LOUPE_ZOOM)
            src = canvas[max(my - r, TOP) : my + r, max(mx - r, 0) : mx + r].copy()
            if src.size:
                lo = cv2.resize(src, (LOUPE, LOUPE), interpolation=cv2.INTER_NEAREST)
                cv2.drawMarker(lo, (LOUPE // 2, LOUPE // 2), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
                lx = mx + 20 if mx + 20 + LOUPE < W_TOTAL else mx - 20 - LOUPE
                ly = int(np.clip(my - LOUPE // 2, TOP, TOP + VH - LOUPE))
                canvas[ly : ly + LOUPE, lx : lx + LOUPE] = lo
                cv2.rectangle(canvas, (lx, ly), (lx + LOUPE - 1, ly + LOUPE - 1), (255, 255, 255), 1)
        # header
        saved = self.lab["labels"].get(self.clip, {}).get(str(self.frame))
        state = "SAVED " + saved["kind"].upper() if saved else "not saved"
        pend = {"tip": "marker set - check it, then Save & Next", "none": "NO TIP - Save & Next",
                None: "click the tip end"}[self.pending_kind]  # fmt: skip
        if self.pending_kind == "tip" and self.multi() > 1 and len(self.pending) < 2:
            pend = "now click the LAST tip"
        multi = f"   MULTICHANNEL x{self.multi()}: first + last tip" if self.multi() > 1 else ""
        head = (f"{self.clip}  frame {self.frame}  [{state}]  {pend}{multi}   "
                f"queue {self.q + 1}/{len(self.queue)}  labels {self.n_labelled()}/{self.target}")  # fmt: skip
        cv2.putText(canvas, head, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # timeline
        T = len(frames)
        ty = TOP + VH + 8
        for f in range(T):
            xa, xb = int(8 + f * (W_TOTAL - 16) / T), int(8 + (f + 1) * (W_TOTAL - 16) / T) - 1
            e = self.lab["labels"].get(self.clip, {}).get(str(f))
            col = (0, 200, 0) if e and e["kind"] == "tip" else (0, 0, 200) if e else (70, 70, 70)
            cv2.rectangle(canvas, (xa, ty), (max(xa + 1, xb), ty + TL_H), col, -1)
        xc = int(8 + (frame + 0.5) * (W_TOTAL - 16) / T)
        cv2.rectangle(canvas, (xc - 2, ty - 3), (xc + 2, ty + TL_H + 3), (255, 255, 255), -1)
        # buttons
        for k, label in BUTTONS:
            bx0, by0, bx1, by1 = self.btn_rects[k]
            hot = self.mouse is not None and bx0 <= self.mouse[0] < bx1 and by0 <= self.mouse[1] < by1
            fill = (60, 140, 60) if k == "next" else (90, 90, 90)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), tuple(min(255, c + 40) for c in fill) if hot else fill, -1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(canvas, label, ((bx0 + bx1 - tw) // 2, (by0 + by1 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
        foot = self.msg or "a/d: step frames in this clip   h: heatmap   g: show dispensed well   q: quit"
        cv2.putText(canvas, foot, (8, H_TOTAL - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 255) if self.msg else (180, 180, 180), 1, cv2.LINE_AA)  # fmt: skip
        return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="9,10,1,2")
    ap.add_argument("--per-clip", type=int, default=8)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-model", action="store_true", help="no heatmap / suggestions")
    ap.add_argument("--snapshot", default=None, help="render one frame to this image and exit (no window)")
    a = ap.parse_args()
    splits = load_splits()
    plates = {int(x) for x in a.plates.split(",") if x}
    if plates & set(splits["test"] + splits["val"]):
        sys.exit("only train plates are labelled (val/test stay untouched for evaluation)")
    clips = split_clips("train", plates)
    tool = LabelTool(build_queue(clips, a.per_clip, np.random.default_rng(0)), clip_wells(), Path(a.out),
                     Model(not a.no_model), a.target)  # fmt: skip
    if a.snapshot:
        cv2.imwrite(a.snapshot, tool.render())
        return
    cv2.namedWindow("tips", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("tips", tool.on_mouse)
    while True:
        cv2.imshow("tips", tool.render())
        k = cv2.waitKey(1000 // PLAY_FPS if tool.play else 30)
        if k != -1 and not tool.key(k & 0xFF):
            break
    tool.write()
    cv2.destroyAllWindows()
    print(f"saved {tool.n_labelled()} labels -> {tool.out}")


if __name__ == "__main__":
    main()
