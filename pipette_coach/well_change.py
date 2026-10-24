"""Well change: did liquid go into a well? Compare the well with itself before and after the tip's visit.

The idea of touch detection on a surface (a finger pressed on glass turns pale, its shadow joins the finger): a
small, local change, measured against the same spot a moment earlier. Here, on the plate warped to a fixed grid (each
well centred in its own cell):
  texture  spread of brightness inside the well. An empty well shows its wall as a bright crescent; liquid fills
           the bottom, so the crescent fades (pipette1, clear water: after/before 0.3-0.75, unchanged wells ~1.0).
  colour   mean chroma (Lab a, b) inside the well; a dye moves it, clear water barely does.
Both are taken relative to the whole plate (median over the other wells), so light or exposure changes cancel out.

Covered wells (hand, sleeve, pipette) are found from the CELL CORNERS only: the plastic between wells, which keeps its
colour whatever goes into the wells. A corner far from its usual look = covered, and that frame is not used for the
well. So a red or blue liquid is not mistaken for a hand. The usual look is learnt only from frames that already
look clear (slow average), so a sleeve resting on the plate for a while is not learnt as "plate"; a spot that has
looked covered for reset_s (plate moved, or a hand there at the start) is re-learnt.

Verdict: liquid if the texture dropped (< tex_ratio) or the colour moved (> color_de); no liquid only if the well
looks the SAME (texture within same_tex, colour within color_de); anything else (e.g. texture up: still covered by
something the corner test missed) = not seen clearly.
"""

from collections import deque

import cv2
import numpy as np

S = 24  # px per well pitch in the warped plate


def _masks(s: int):
    yy, xx = np.mgrid[:s, :s] - (s - 1) / 2
    r = np.hypot(xx, yy) / s
    return r <= 0.33, r >= 0.6  # inside the well; the 4 cell corners (plastic between wells)


def well_cells(frame: np.ndarray, H: np.ndarray, s: int = S) -> np.ndarray:
    """The plate warped so well (column c, row r) is centred in the cell [r*s, (r+1)*s) x [c*s, (c+1)*s)."""
    A = np.array([[s, 0, s / 2], [0, s, s / 2], [0, 0, 1]], np.float64)
    return cv2.warpPerspective(frame, A @ np.linalg.inv(H), (12 * s, 8 * s), flags=cv2.INTER_LINEAR)


def cell_features(cells: np.ndarray, s: int = S) -> np.ndarray:
    """(8, 12, 7) per well: texture (std of L inside the well), mean Lab inside the well, mean Lab of the corners."""
    lab = cv2.cvtColor(cells, cv2.COLOR_BGR2LAB).astype(np.float32)
    c = lab.reshape(8, s, 12, s, 3).transpose(0, 2, 1, 3, 4).reshape(8, 12, s * s, 3)
    disc, corner = (m.ravel() for m in _masks(s))
    d = c[:, :, disc]
    return np.concatenate([d[..., :1].std(2), d.mean(2), c[:, :, corner].mean(2)], -1)


class WellChange:
    """Keeps per-well features of the last history_s seconds; score(well, t0, t1) compares the well after the
    visit [t0, t1] with before it.

    update(t, frame, H) every frame (or update_features with precomputed cell_features). decide() gives the verdict
    as soon as the well has been seen clearly for early_clean frames after the tip left (usually a fraction of a
    second) and the answer is clear; it waits up to post[1] s while the well is covered or the answer is borderline."""

    def __init__(self, history_s: float = 30.0, cover_de: float = 25.0, pre=(5.0, 0.3), post=(0.0, 4.0),
                 min_clean: int = 6, early_clean: int = 3, early_tex: float = 0.7, early_de: float = 8.0,
                 same_after_s: float = 1.5, same_clean: int = 15, tex_ratio: float = 0.8, color_de: float = 5.0,
                 same_tex=(0.87, 1.08), ref_rate: float = 0.02, reset_s: float = 20.0, s: int = S):  # fmt: skip
        self.history_s, self.cover_de, self.pre, self.post = history_s, cover_de, pre, post
        self.min_clean, self.early_clean, self.tex_ratio = min_clean, early_clean, tex_ratio
        self.early_tex, self.early_de = early_tex, early_de
        self.same_after_s, self.same_clean = same_after_s, same_clean
        self.color_de, self.same_tex, self.ref_rate, self.reset_s, self.s = color_de, same_tex, ref_rate, reset_s, s
        self.t: deque = deque()
        self.f: deque = deque()
        self.clean: deque = deque()
        self.ref = None  # usual corner colour per well
        self.covered_since = None
        self._memo = (None, None)
        self._before: dict = {}

    def update(self, t: float, frame: np.ndarray, H) -> None:
        if H is None or not np.isfinite(H).all():
            return
        self.update_features(t, cell_features(well_cells(frame, H, self.s), self.s))

    def update_features(self, t: float, f: np.ndarray) -> None:
        corner = f[..., 4:]
        if self.ref is None:
            self.ref, self.covered_since = corner.copy(), np.full((8, 12), t)
        clean = np.linalg.norm(corner - self.ref, axis=-1) <= self.cover_de
        self.ref[clean] += self.ref_rate * (corner[clean] - self.ref[clean])
        self.covered_since[clean] = t
        stale = t - self.covered_since > self.reset_s
        self.ref[stale], self.covered_since[stale] = corner[stale], t
        self.t.append(t), self.f.append(f), self.clean.append(clean)
        while self.t and self.t[0] < t - self.history_s:
            self.t.popleft(), self.f.popleft(), self.clean.popleft()

    def _medians(self, T, F, C, a: float, b: float, n_min: int) -> tuple[np.ndarray, np.ndarray]:
        m = (T >= a) & (T <= b)
        out = np.full((8, 12, 4), np.nan, np.float32)
        Fm, Cm = F[m][..., :4], C[m]
        n = Cm.sum(0)
        for r, c in zip(*np.nonzero(n >= n_min), strict=True):
            out[r, c] = np.median(Fm[Cm[:, r, c], r, c], 0)
        return out, n

    def _recent(self, since: float):
        """Times, features and clear flags of the frames from `since` on (the newest few seconds only)."""
        k = 0
        while k < len(self.t) and self.t[-1 - k] >= since:
            k += 1
        if not k:
            return np.zeros(0), np.zeros((0, 8, 12, 7), np.float32), np.zeros((0, 8, 12), bool)
        idx = range(len(self.t) - k, len(self.t))
        return (np.array([self.t[i] for i in idx]), np.array([self.f[i] for i in idx]),
                np.array([self.clean[i] for i in idx]))  # fmt: skip

    def _compare(self, t0: float, t1: float, t_end: float, n_min: int):
        """Texture ratio and chroma shift of every well, after [t1 + post[0], t_end] vs before the visit (memoised:
        many wells are asked about the same window)."""
        key = (t0, t1, t_end, n_min, self.t[-1])
        if self._memo[0] == key:
            return self._memo[1]
        if t0 not in self._before:  # the before-window is in the past: computed once per visit
            if len(self._before) > 64:
                self._before.clear()
            self._before[t0] = self._medians(*self._recent(t0 - self.pre[0]), t0 - self.pre[0], t0 - self.pre[1],
                                             self.min_clean)[0]  # fmt: skip
        a, n_after = self._medians(*self._recent(t1 + self.post[0]), t1 + self.post[0], t_end, n_min)
        b = self._before[t0]
        res = (a[..., 0] / np.maximum(b[..., 0], 1e-3), a[..., 2:] - b[..., 2:], n_after)  # chroma (Lab a, b) only
        self._memo = (key, res)
        return res

    def score(self, well, t0: float, t1: float, skip=(), t_now: float | None = None, n_min: int | None = None):
        """well = (column, row). Compares the well after the visit (up to t_now, at most post[1] s) with before.
        Without t_now: None until post[1] s after t1 have been seen. Otherwise a dict: tex = texture after/before
        relative to the plate, de = colour shift relative to the plate, filled = True if tex < tex_ratio or
        de > color_de; False if the well looks the same; None = not seen clearly enough before or after (or
        borderline). skip: other wells to leave out of the plate reference (e.g. wells filled in the same windows)."""
        if t_now is None:
            if not self.t or self.t[-1] < t1 + self.post[1]:
                return None
            t_now = t1 + self.post[1]
        tex, shift, n_after = self._compare(t0, t1, min(t_now, t1 + self.post[1]), n_min or self.min_clean)
        ref = np.isfinite(tex)
        c, r = well
        ref[r, c] = False
        for cc, rr in skip:
            ref[rr, cc] = False
        if ref.sum() < 10 or not np.isfinite(tex[r, c]):
            return {"tex": float("nan"), "de": float("nan"), "filled": None, "n_after": int(n_after[r, c])}
        t_rel = tex[r, c] / np.median(tex[ref])
        de = float(np.linalg.norm(shift[r, c] - np.median(shift[ref], 0)))
        if t_rel < self.tex_ratio or de > self.color_de:
            filled = True
        elif self.same_tex[0] <= t_rel <= self.same_tex[1]:
            filled = False
        else:
            filled = None
        return {"tex": float(t_rel), "de": de, "filled": filled, "n_after": int(n_after[r, c])}

    def decide(self, well, t0: float, t1: float, t: float, skip=()) -> dict | None:
        """Verdict at time t for the visit [t0, t1], or None = keep waiting. Asymmetric, so "liquid" shows at once:
        liquid     as soon as early_clean clear frames after the tip left show an unmistakable change
                   (texture < early_tex or colour > early_de; pipette1: real dispenses 0.35-0.65 within ~0.1 s)
        any        from same_after_s on, with same_clean clear frames (the normal thresholds; "no liquid" only
                   here: a well still half hidden right after the tip left can look unchanged)
        final      at t1 + post[1], whatever there is (filled None = not seen clearly)."""
        if not self.t or t < t1 + self.post[0]:
            return None
        if t >= t1 + self.post[1]:
            return {**self.score(well, t0, t1, skip, t_now=t), "early": False}
        r = self.score(well, t0, t1, skip, t_now=t, n_min=self.early_clean)
        if r["filled"] and (r["tex"] < self.early_tex or r["de"] > self.early_de):
            return {**r, "early": True}
        if t >= t1 + self.same_after_s:
            r = self.score(well, t0, t1, skip, t_now=t, n_min=self.same_clean)
            if r["filled"] is not None:
                return {**r, "early": True}
        return None


class DispenseCheck:
    """A verdict for each logged dispense from the well change: liquid seen / no liquid / not seen clearly.

    watch(idx, well, t0) when the dwell rule logs event idx (call again with the new well when it is revised).
    update(t, frame, H, tip_well) every frame: the visit lasts while the tip comes back to the well; once the tip has
    left, the event is decided as soon as the well is seen clearly (WellChange.decide; at most post[1] s).
    Returns [(idx, score dict)]: dict["filled"] True = liquid seen,
    False = the well did not change (a hover, not a dispense), None = not seen clearly (keep the dwell decision)."""

    def __init__(self, **wc_kw):
        self.wc = WellChange(**wc_kw)
        self.pending: dict = {}

    def watch(self, idx: int, well, t0: float) -> None:
        self.pending[idx] = {"well": tuple(well), "t0": t0, "t1": t0}

    def update(self, t: float, frame: np.ndarray, H, tip_well) -> list:
        self.wc.update(t, frame, H)
        out = []
        for idx, p in list(self.pending.items()):
            if tip_well is not None and tuple(tip_well) == p["well"]:
                p["t1"] = t
                continue
            skip = [q["well"] for j, q in self.pending.items() if j != idx]
            r = self.wc.decide(p["well"], p["t0"], p["t1"], t, skip=skip)
            if r is not None:
                out.append((idx, {**r, "after_s": round(t - p["t1"], 2)}))
                del self.pending[idx]
        return out
