"""Self-check + human review sheet for per-clip plate grids (train references / grid-verifier labels).

For each clip:
  A = stored grid (data/processed/clip_grids.json), B = current pipeline grid (session mode + verifier).
  Auto-check of A: verifier vote over 6 frames (is A preferred over its half-well / row / column shifts, by what
  margin), ring support (share of A's visible wells with a ring under them), and agreement with B. Combined into a
  confidence and a suggested answer.
Clips are sorted by confidence (lowest first); pages show the frame with the most visible wells, A and (if it differs)
B. The reviewer answers per clip: A, B, or neither (-> scripts/label_corners.py). Output stays local:
runs/<ts>_grid-review/{page_*.jpg, review.md, grids.json}

    python scripts/make_grid_review.py --plates 1,2,9,10
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.data import PROCESSED_DIR, clips_for_split, load_splits, plate_of, plate_rows_toward  # noqa: E402
from pipette_coach.localizer import localize_clip_frames  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector  # noqa: E402
from pipette_coach.localizer.fusion import well_observations  # noqa: E402
from pipette_coach.localizer.geometry import GRID, OUTLINE, grid_error, project  # noqa: E402
from pipette_coach.localizer.rings import find_well_candidates  # noqa: E402
from pipette_coach.localizer.session import read_frames, session_prior  # noqa: E402
from pipette_coach.localizer.verifier import OFFSETS, GridVerifier, offset_H  # noqa: E402
from pipette_coach.runs import new_run_dir  # noqa: E402

Z = 4  # zoom
TILE = (720, 520)


def draw(frame: np.ndarray, H: np.ndarray, title: str, color: tuple) -> np.ndarray:
    q = project(H, OUTLINE)
    x0, y0 = np.floor(q.min(0) - 8).astype(int)
    x1, y1 = np.ceil(q.max(0) + 8).astype(int)
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)
    for k, (x, y) in enumerate(project(H, GRID)):
        X, Y = (x - x0 + 0.5) * Z - 0.5, (y - y0 + 0.5) * Z - 0.5
        if 0 <= X < crop.shape[1] and 0 <= Y < crop.shape[0]:
            c = (255, 0, 255) if k == 0 else color
            cv2.drawMarker(crop, (int(round(X)), int(round(Y))), c, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    s = min(TILE[0] / crop.shape[1], (TILE[1] - 40) / crop.shape[0])
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    tile = np.zeros((TILE[1], TILE[0], 3), np.uint8)
    tile[40 : 40 + crop.shape[0], : crop.shape[1]] = crop
    cv2.putText(tile, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return tile


def auto_check(frames: list, A: np.ndarray, B: np.ndarray | None, cfg: dict, ver: GridVerifier) -> dict:
    """Confidence of the stored grid A from three independent cues."""
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    top, margins = 0, []
    for i in idx:
        s = ver.scores(frames[i], [offset_H(A, dx, dy) for dx, dy in OFFSETS])
        top += int(np.argmax(s) == 0)
        margins.append(float(s[0] - np.max(s[1:])))
    f = frames[len(frames) // 2]
    P, _, _ = find_well_candidates(f, cfg)
    g = project(A, GRID)
    sup = float(np.mean(np.linalg.norm(g[:, None] - P[None], axis=2).min(1) < 0.3 * 14.5)) if len(P) else 0.0
    agree = B is None or grid_error(B, A) <= 0.25
    conf = (top / len(idx)) * 0.5 + max(0.0, min(1.0, float(np.median(margins)))) * 0.3 + sup * 0.2
    if top / len(idx) >= 0.8 and agree:
        suggest = "A"
    elif top / len(idx) <= 0.2 and not agree:
        suggest = "B?"
    else:
        suggest = "check"
    return {
        "verifier_top": f"{top}/{len(idx)}",
        "verifier_margin": float(np.median(margins)),
        "ring_support": sup,
        "agrees_with_B": agree,
        "confidence": conf,
        "suggest": suggest,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", default="1,2")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--per-page", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config("localizer")
    det = PlateDetector(cfg)
    ver = GridVerifier(REPO_ROOT / cfg["verifier"]["weights"])
    splits = load_splits(a.splits)
    grids = json.loads((PROCESSED_DIR / "clip_grids.json").read_text())
    run = new_run_dir("grid-review")
    items = []
    for pl in [int(x) for x in a.plates.split(",") if x]:
        if pl not in splits["train"]:
            sys.exit(f"plate {pl} is not a train plate; only train references are reviewed here")
        clips = [c for c in clips_for_split("train", a.splits) if plate_of(c) == pl]
        prior = session_prior(clips, cfg, det)
        for clip in clips:
            frames = read_frames(clip)
            fits = localize_clip_frames(
                frames, cfg, det, prior_H=prior, rows_toward=plate_rows_toward(pl), verifier=ver
            )
            quads = [(d[0] if (d := det.detect(f)) is not None else None) for f in frames[::4]]
            counts = [
                len(well_observations(f, q, cfg)[0]) if q is not None else 0
                for f, q in zip(frames[::4], quads, strict=True)
            ]
            i = 4 * int(np.argmax(counts))  # frame with the most visible wells
            A = np.array(grids[clip]["H"]) if grids.get(clip, {}).get("H") else None
            B = fits[i].H if fits[i].found else None
            if A is None and B is None:
                continue
            chk = auto_check(frames, A, B, cfg, ver) if A is not None else {"confidence": -1.0, "suggest": "B?"}
            tiles = []
            if A is not None:
                tiles.append(draw(frames[i], A, f"{clip}  A (stored)  conf {chk['confidence']:.2f}", (0, 255, 0)))
            show_b = B is not None and (A is None or grid_error(B, A) > 0.25)
            if show_b:
                tiles.append(draw(frames[i], B, f"{clip}  B (pipeline)", (0, 140, 255)))
            while len(tiles) < 2:
                tiles.append(np.zeros_like(tiles[0]))
            items.append(
                {
                    "clip": clip,
                    "frame": i,
                    "row": np.hstack(tiles),
                    "check": chk,
                    "show_b": show_b,
                    "A": None if A is None else A.tolist(),
                    "B": None if B is None else B.tolist(),
                }
            )
            print(clip, chk.get("suggest"), round(chk["confidence"], 2), flush=True)
    items.sort(key=lambda x: x["check"]["confidence"])  # least confident first
    rows = [it["row"] for it in items]
    md = [
        "# Grid review (least confident first)",
        "",
        "Reply per clip: A, B, or neither (neither -> `python scripts/label_corners.py CLIP`). Magenta cross = A1.",
        "suggest: A = auto-check trusts A; B? = A looks wrong and B may be right; check = unsure.",
        "",
        "| # | clip | page | options | suggest | verifier picks A | margin | ring support | A agrees with B |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n, it in enumerate(items):
        c = it["check"]
        md.append(
            f"| {n + 1} | {it['clip']} | {n // a.per_page + 1} | {'A / B' if it['show_b'] else 'A'} | {c['suggest']} | "
            f"{c.get('verifier_top', '-')} | {c.get('verifier_margin', 0):.2f} | {c.get('ring_support', 0):.0%} | "
            f"{c.get('agrees_with_B', '-')} |"
        )
    (run / "grids.json").write_text(
        json.dumps({it["clip"]: {"A": it["A"], "B": it["B"], "frame": it["frame"]} for it in items})
    )
    for p_i in range(0, len(rows), a.per_page):
        cv2.imwrite(
            str(run / f"page_{p_i // a.per_page + 1}.jpg"),
            np.vstack(rows[p_i : p_i + a.per_page]),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    md += ["", "Contains dataset frames: keep local."]
    (run / "review.md").write_text("\n".join(md) + "\n")
    print("review sheet:", run)


if __name__ == "__main__":
    main()
