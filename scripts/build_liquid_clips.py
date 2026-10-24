"""Cut a short clip around every SPACE mark of a dispense recording (scripts/record_dispenses.py) - the training
material for the liquid-release model. Only these windows are read, not the whole video.

Window: mark frame +- --half-s seconds (default 1 s: the press lands a reaction time after the real release, and
the window also holds the moments just before / after it). Output: data/processed/liquid_clips/<recording>/
  mark_<k>.mp4      the window, original size and fps
  clips.json        per clip: mark frame / time, first and last frame of the window in the full video

    python scripts/build_liquid_clips.py --rec data/own/pipette8
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipette_coach.data import PROCESSED_DIR  # noqa: E402


def windows(marks: list[dict], fps: float, n_frames: int, half_s: float) -> list[dict]:
    """[{"k", "mark_frame", "mark_t", "f0", "f1"}]: f0..f1 (inclusive) = mark frame +- half_s, clipped to the video."""
    h = round(half_s * fps)
    return [{"k": k, "mark_frame": m["frame"], "mark_t": m["t"], "f0": max(0, m["frame"] - h),
             "f1": min(n_frames - 1, m["frame"] + h)} for k, m in enumerate(marks)]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, help="recording folder with video.mp4 + dispenses.json")
    ap.add_argument("--half-s", type=float, default=1.0)
    a = ap.parse_args()
    rec = Path(a.rec)
    marks = json.loads((rec / "dispenses.json").read_text())["dispenses"]
    cap = cv2.VideoCapture(str(rec / "video.mp4"))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = PROCESSED_DIR / "liquid_clips" / rec.name
    out.mkdir(parents=True, exist_ok=True)
    wins = windows(marks, fps, n, a.half_s)
    for w in wins:
        cap.set(cv2.CAP_PROP_POS_FRAMES, w["f0"])
        wr = cv2.VideoWriter(str(out / f"mark_{w['k']:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for _ in range(w["f1"] - w["f0"] + 1):
            ok, f = cap.read()
            if not ok:
                break
            wr.write(f)
        wr.release()
    (out / "clips.json").write_text(json.dumps({"video": str(rec / "video.mp4"), "fps": fps, "half_s": a.half_s,
                                                "clips": wins}, indent=1))  # fmt: skip
    print(f"{len(wins)} clips of +-{a.half_s} s -> {out}")


if __name__ == "__main__":
    main()
