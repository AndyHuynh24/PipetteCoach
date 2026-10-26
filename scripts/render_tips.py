"""Live tip -> well -> dispense pipeline on a video, frame by frame as from a camera, rendered.

Per frame: PlateTracker (plate + 12 x 8 grid) -> plate crop (outline + 25 %) -> tip model (most confident tip >=
conf) -> tip end -> well through the grid (pipette_coach.tips.well_of) -> DwellDetector (same well >= 0.4 s = one
candidate) -> liquid check (counted only if liquid shows in the well). Drawn: plate grid, tip end (cyan), its well
(ring filling up while the tip stays), dispense log, zoom.
Output: runs/<ts>_tips-<name>/{tips.mp4, events.json, metrics.json, notes.md}. Keep local (third-party video).

    python scripts/render_tips.py --video data/raw/pipette1.mp4 --start 340 --seconds 180 \
        --model models/tip_all/weights/best.pt --conf 0.5
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
from render_all import encode  # noqa: E402
from render_plate_rebuild import draw  # noqa: E402
from run_live import add_zoom, draw_log  # noqa: E402

from pipette_coach.config import REPO_ROOT, TIP_MODEL, load_config  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector, _default_device  # noqa: E402
from pipette_coach.localizer.geometry import GRID, pitch_px, project  # noqa: E402
from pipette_coach.localizer.live import PlateTracker  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tips import DwellDetector, plate_crop_box, well_of  # noqa: E402
from pipette_coach.well_change import DispenseCheck  # noqa: E402


def well_name(w) -> str:
    return f"{'ABCDEFGH'[w[1]]}{w[0] + 1}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--seconds", type=float, default=60, help="0 = to the end")
    ap.add_argument("--model", default=TIP_MODEL)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--track-every", type=int, default=2)
    ap.add_argument("--name", default="pipette1")
    ap.add_argument("--dwell", default="{}", help='DwellDetector settings as JSON, e.g. {"min_s": 0.2}')
    ap.add_argument("--no-well-change", action="store_true", help="no liquid check (pipette_coach/well_change.py)")
    ap.add_argument(
        "--rows-toward",
        choices=["down", "up", "left", "right"],
        default=None,
        help="image direction of rows A -> H (as live_demo)",
    )
    a = ap.parse_args()
    from ultralytics import YOLO

    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    tip_model, dev = YOLO(a.model), _default_device()
    tracker = PlateTracker(
        cfg, det, net, rows_toward=a.rows_toward, track_every=a.track_every, verify_every=30
    )  # as live_demo
    check = None if a.no_well_change else DispenseCheck()
    dwell = DwellDetector(**{"trip_gap_s": None if check else 1.0, **json.loads(a.dwell)})  # as live_demo
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
    n = int(a.seconds * fps) if a.seconds > 0 else 10**9
    run = new_run_dir(f"tips-{a.name}")
    vw, events, ms = None, [], []
    tracker.update(np.zeros((1080, 1920, 3), np.uint8))  # warm-up
    tip_model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=a.imgsz, verbose=False, device=dev)
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        t = a.start + i / fps
        t0 = time.perf_counter()
        out = tracker.update(f)
        tip, well = None, None
        if out.get("outline") is not None:
            x0, y0, x1, y1 = plate_crop_box(np.asarray(out["outline"]), 0.25, f.shape[1], f.shape[0])
            r = tip_model.predict(f[y0:y1, x0:x1], imgsz=a.imgsz, conf=a.conf, verbose=False, device=dev)[0]
            if len(r.boxes):
                k = int(r.boxes.conf.argmax())
                tip = r.keypoints.xy.cpu().numpy()[k, 0] + [x0, y0]
                well = well_of(out["H"], tip)
        ev = dwell.update(t, well)
        if ev is not None:
            if dwell.revised:  # a longer stop in the same trip: that was the dispense, not the pause before it
                print(f"{t:7.2f}s dispense {events[-1]['well']} -> {well_name(ev)}", flush=True)
                events[-1] = {**events[-1], "well": well_name(ev), "revised_from": events[-1]["well"]}
            else:
                events.append({"t": round(t, 2), "well": well_name(ev)})
                print(f"{t:7.2f}s dispense {well_name(ev)}", flush=True)
            if check:
                check.watch(len(events) - 1, ev, dwell.start)
        for idx, r in check.update(t, f, out.get("H"), well) if check else []:
            events[idx].update(liquid=r["filled"], tex=round(r["tex"], 2), de=round(r["de"], 1), after_s=r["after_s"])
            print(f"{t:7.2f}s {events[idx]['well']}: liquid {r['filled']} (texture {r['tex']:.2f}, "
                  f"colour {r['de']:.1f})", flush=True)  # fmt: skip
        ms.append((time.perf_counter() - t0) * 1000)
        # ---- drawing (not timed)
        img = draw(f, out["box"], out if out["H"] is not None else None, out.get("wells"),
                   f"t={t:6.1f}s  {out['status']}   {ms[-1]:.0f} ms")  # fmt: skip
        if tip is not None:
            lw = max(2, round(f.shape[1] / 600))
            if well is not None:
                c = project(out["H"], GRID[[well[1] * 12 + well[0]]])[0]
                rad = int(pitch_px(out["H"]) * 0.5)
                prog = dwell.progress(t)
                col = (0, 0, 255) if prog >= 1 else (0, 200, 255)
                cv2.ellipse(img, tuple(np.round(c).astype(int)), (rad, rad), -90, 0, 360 * prog, col, 2 * lw)
                cv2.putText(img, well_name(well), (int(c[0]) + rad, int(c[1]) - rad), 0, 1.0, col, 2, cv2.LINE_AA)
            cv2.drawMarker(img, tuple(np.round(tip).astype(int)), (255, 255, 0), cv2.MARKER_CROSS, 24, lw)
        img = add_zoom(img, out)
        draw_log(img, events)
        if vw is None:
            vw = cv2.VideoWriter(str(run / "raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
        vw.write(img)
    vw.release()
    encode(run / "raw.mp4", run / "tips.mp4")
    ms = np.array(ms)
    m = {"video": a.video, "start": a.start, "frames": len(ms), "model": a.model, "conf": a.conf, "dwell": a.dwell,
         "well_change": not a.no_well_change,
         "ms_p50": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)), "fps_mean": float(1000 / ms.mean()),
         "dispenses": len(events), "no_liquid": sum(e.get("liquid") is False for e in events),
         "liquid_not_seen": sum("liquid" in e and e["liquid"] is None for e in events)}  # fmt: skip
    write_metrics(run, m)
    (run / "events.json").write_text(json.dumps(events, indent=1))
    (run / "notes.md").write_text(f"# Tip -> well -> dispense ({a.video} from {a.start}s)\n\n{m}\n\nKeep local.\n")
    print(m, "->", run)


if __name__ == "__main__":
    main()
