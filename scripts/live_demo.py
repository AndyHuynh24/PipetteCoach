"""Live demo: webcam (or a video played at real speed, like a camera) -> plate grid -> tip -> well -> dispense log.

Threads (pipette_coach/live_pipeline.py): camera reader (newest frame only), plate tracker (as often as it can),
processing = tip model + well + dwell + drawing on EVERY frame it gets; the main thread only shows the newest
result (window). Measured: processed frames per second, share of camera frames processed (rest dropped because
processing was busy), plate updates per second, age of the grid in frames, window frames per second.

    python scripts/live_demo.py --camera 0 --show                          # live webcam window (q to quit)
    python scripts/live_demo.py --camera 0 --show --rows-toward down       # plate with rows A -> H going down
    python scripts/live_demo.py --video data/raw/pipette1.mp4 --start 300 --seconds 60   # benchmark, no window
    python scripts/live_demo.py --video data/raw/pipette1.mp4 --start 300 --seconds 60 --save
    python scripts/live_demo.py --camera 0 --show --ui http://localhost:7500   # + Well Plate UI (wellPlateUiWeb/)
    python scripts/live_demo.py --camera http://192.168.1.20:8080/video --show --ui http://localhost:7500   # phone app
    python scripts/list_cameras.py                                         # which index is which camera (e.g. a phone)

With --ui, every dispense whose liquid check says "liquid" is sent to the Well Plate UI (POST /api/drop, from a
background thread), the well under the tip is sent whenever it changes (POST /api/camera/tip: the page rings it green
or red), and the A1 corner set on the page sets the row direction unless --rows-toward is given.
Token: WELLPLATE_API_TOKEN, if the server uses one.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_plate_rebuild import draw  # noqa: E402
from run_live import add_zoom, draw_log  # noqa: E402

from pipette_coach.config import REPO_ROOT, load_config  # noqa: E402
from pipette_coach.live_pipeline import LatestFrame, PlateWorker, TipFinder, serialize_gpu  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector, _default_device  # noqa: E402
from pipette_coach.localizer.geometry import GRID, pitch_px, project  # noqa: E402
from pipette_coach.localizer.live import PlateTracker  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tips import DwellDetector  # noqa: E402
from pipette_coach.ui_link import UiLink  # noqa: E402
from pipette_coach.well_change import DispenseCheck  # noqa: E402


def well_name(w) -> str:
    return f"{'ABCDEFGH'[w[1]]}{w[0] + 1}"


def overlay(frame, plate, tip, well, dwell, t, events, fps_shown):
    img = draw(frame, plate.get("box"), plate if plate.get("H") is not None else None, plate.get("wells"),
               f"t={t:6.1f}s  {plate['status']}   {fps_shown:4.1f} fps")  # fmt: skip
    if tip is not None:
        lw = max(2, round(frame.shape[1] / 600))
        if well is not None:
            c = project(plate["H"], GRID[[well[1] * 12 + well[0]]])[0]
            rad = int(pitch_px(plate["H"]) * 0.5)
            prog = dwell.progress(t)
            col = (0, 0, 255) if prog >= 1 else (0, 200, 255)
            cv2.ellipse(img, tuple(np.round(c).astype(int)), (rad, rad), -90, 0, 360 * prog, col, 2 * lw)
            cv2.putText(img, well_name(well), (int(c[0]) + rad, int(c[1]) - rad), 0, 1.0, col, 2, cv2.LINE_AA)
        cv2.drawMarker(img, tuple(np.round(tip).astype(int)), (255, 255, 0), cv2.MARKER_CROSS, 24, lw)
    img = add_zoom(img, plate)
    draw_log(img, events)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=None, help="camera index (0, 1, ...) or a phone app's stream URL")
    ap.add_argument("--video", default=None)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until the end / q)")
    ap.add_argument("--model", default="models/tip_final/weights/last.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--show", action="store_true", help="show a window (q quits)")
    ap.add_argument("--save", action="store_true", help="also write the shown frames to live.mp4")
    ap.add_argument("--name", default="live-demo")
    ap.add_argument("--no-well-change", action="store_true", help="no liquid check (pipette_coach/well_change.py)")
    ap.add_argument(
        "--rows-toward",
        choices=["down", "up", "left", "right"],
        default=None,
        help="image direction of rows A -> H (default: A1 = corner nearest the image top-left)",
    )
    ap.add_argument("--size", default="1920x1080", help="resolution to ask the camera for (WxH)")
    ap.add_argument("--ui", default=None, help="Well Plate UI address, e.g. http://localhost:7500")
    a = ap.parse_args()
    if a.ui and a.no_well_change:
        ap.error("--ui sends only dispenses with liquid seen, so it needs the liquid check")
    from ultralytics import YOLO

    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    dev = _default_device()
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / "models" / "well_net" / "best.pt")
    # periodic full check (corrects e.g. a grid re-locked one column off from a partial view): ~1.5 s at ~20 Hz;
    # it runs in the tracker thread, so it does not slow the video
    ui = UiLink(a.ui, os.environ.get("WELLPLATE_API_TOKEN")).start() if a.ui else None
    if ui and ui.errors:
        print(f"Well Plate UI not reachable yet ({ui.errors[-1]}); dispenses are sent once it is", flush=True)
    tracker = PlateTracker(cfg, det, net, rows_toward=a.rows_toward or (ui and ui.rows_toward), verify_every=30)
    tip = TipFinder(YOLO(a.model), conf=a.conf, imgsz=a.imgsz, device=dev)
    serialize_gpu(det, net, tip.model)  # tracker thread + main loop share the GPU: one model call at a time
    # warm-up (first calls compile kernels) before the clock starts
    tracker.update(np.zeros((1080, 1920, 3), np.uint8))
    tracker.reset()
    tip.model.predict(np.zeros((480, 480, 3), np.uint8), imgsz=a.imgsz, verbose=False, device=dev)

    if a.camera is not None and a.camera.isdigit():
        a.camera = int(a.camera)
    src = a.camera if a.camera is not None else a.video
    cam = LatestFrame(src, start_s=a.start if a.video else 0.0, size=tuple(int(v) for v in a.size.split("x")))
    if a.camera is not None:
        cam.next(-1)  # the first frame tells the real size
        print(f"camera {a.camera}: {cam.size[0]}x{cam.size[1]} at {cam.fps:.0f} fps", flush=True)
    plate = PlateWorker(tracker, cam)
    check = None if a.no_well_change else DispenseCheck()
    # with the liquid check every stop >= 0.4 s is its own candidate and counts only if liquid shows (several wells
    # in one pass work); without it, one dispense per trip (the longest stop)
    dwell, events = DwellDetector(trip_gap_s=None if check else 1.0), []
    run = new_run_dir(a.name)
    # processing runs in its own thread; the main thread only shows the newest result (macOS allows windows on
    # the main thread only, and showing a frame costs ~15-20 ms - it must not hold up processing)
    st = {"img": None, "img_idx": -1, "shown": 0, "loop_ms": [], "grid_age": [], "t_first": None, "fps": 0.0,
          "stop": False, "vw": None, "why": None}  # fmt: skip

    def process() -> None:
        last = -1
        while not st["stop"]:
            got = cam.next(last)
            if got is None:  # no new frame yet (timeout) or the source ended
                if cam.done:
                    st["why"] = st["why"] or f"source ended ({cam.end_reason})"
                    break
                continue
            last, t, frame = got
            t = t + a.start
            t0 = time.perf_counter()
            st["t_first"] = st["t_first"] or t0
            if ui and not a.rows_toward and tracker.rows_toward != ui.rows_toward:  # A1 corner changed on the page
                tracker.rows_toward = ui.rows_toward
                print(f"{t:7.2f}s rows A -> H now {ui.rows_toward or 'auto'} (from the Well Plate UI)", flush=True)
            p = plate.latest()
            tp, well = tip(frame, p)
            if ui:
                ui.tip(well_name(well) if well is not None else None)  # live ring on the page (sent on change only)
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
            for idx, r in check.update(t, frame, p.get("H"), well) if check else []:  # liquid seen in the well?
                events[idx].update(liquid=r["filled"], tex=round(r["tex"], 2), de=round(r["de"], 1),
                                   after_s=r["after_s"])  # fmt: skip
                print(f"{t:7.2f}s {events[idx]['well']}: liquid {r['filled']} ({r['after_s']:.2f}s after the tip left)",
                      flush=True)  # fmt: skip
                if ui and r["filled"] is True:
                    ui.report(events[idx]["well"])
            img = overlay(frame, p, tp, well, dwell, t, events, st["fps"])
            if a.save:
                if st["vw"] is None:
                    st["vw"] = cv2.VideoWriter(str(run / "live.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cam.fps,
                                               img.shape[1::-1])  # fmt: skip
                st["vw"].write(img)
            st["img"], st["img_idx"] = img, last
            st["shown"] += 1
            st["loop_ms"].append((time.perf_counter() - t0) * 1000)
            if p.get("frame_idx") is not None:
                st["grid_age"].append(last - p["frame_idx"])
            st["fps"] = st["shown"] / max(time.perf_counter() - st["t_first"], 1e-6)
            if a.seconds and t - a.start >= a.seconds:
                st["why"] = st["why"] or "time limit"
                break
        st["why"] = st["why"] or "stop flag"
        st["stop"] = True

    worker = threading.Thread(target=process, daemon=True)
    worker.start()
    displayed = 0
    try:
        if a.show:
            shown_idx = -1
            while not st["stop"]:
                if st["img_idx"] != shown_idx:
                    shown_idx = st["img_idx"]
                    cv2.imshow("Pipette Coach", st["img"])
                    displayed += 1
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    st["why"] = "q pressed"
                    st["stop"] = True
        worker.join()
    finally:
        st["stop"] = True
        worker.join(timeout=5)
        elapsed = time.perf_counter() - (st["t_first"] or time.perf_counter())
        plate.stop()
        cam.stop()
        if ui:
            ui.stop()
        if st["vw"] is not None:
            st["vw"].release()
        cv2.destroyAllWindows()
    shown, loop_ms, grid_age = st["shown"], st["loop_ms"], st["grid_age"]
    lm = np.array(loop_ms) if loop_ms else np.zeros(1)
    m = {"source": str(src), "start": a.start, "model": a.model, "imgsz": a.imgsz, "device": dev,
         "camera_fps": cam.fps, "seconds": round(elapsed, 1), "camera_frames": cam.read_count,
         "processed_frames": shown,
         "processed_fps": shown / max(elapsed, 1e-6), "processed_share": shown / max(cam.read_count, 1),
         "loop_ms_p50": float(np.median(lm)), "loop_ms_p95": float(np.percentile(lm, 95)),
         "plate_updates_per_s": plate.updates / max(elapsed, 1e-6),
         "grid_age_frames_p50": float(np.median(grid_age)) if grid_age else None,
         "grid_age_frames_p95": float(np.percentile(grid_age, 95)) if grid_age else None,
         "window_frames_per_s": displayed / max(elapsed, 1e-6) if a.show else None,
         "stopped_because": st["why"], "dispenses": events, "saved_video": bool(a.save),
         "ui": None if ui is None else {"url": a.ui, "sent": ui.sent, "errors": ui.errors[-20:]}}  # fmt: skip
    write_metrics(run, m)
    (run / "plate_log.json").write_text(json.dumps(plate.log))
    (run / "notes.md").write_text(f"# Live demo ({src})\n\n```\n{json.dumps(m, indent=1)}\n```\n\nKeep local.\n")
    print(json.dumps({k: v for k, v in m.items() if k != "dispenses"}, indent=1), "\n->", run)


if __name__ == "__main__":
    main()
