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

Dispense (default, --liquid none): the tip stays on the same well for --dwell-s (1.5 s) = dispensed there. No liquid
model. Optional liquid checks (--liquid): rule = before/after check once the tip has left (well_change.py); model =
the one-well model decides while liquid goes into the well under the tip (pipette_coach/liquid_live.py).
With --ui, every dispense (with a liquid check: whose check says "liquid") is sent to the Well Plate UI (POST
/api/drop, from a background thread), the well under the tip is sent whenever it changes (POST /api/camera/tip: the
page rings it green or red), and the A1 corner set on the page sets the row direction unless --rows-toward is given.
Token: WELLPLATE_API_TOKEN, if the server uses one.
Record each run (the page's Camera & test > "Record each run"): from Start run until the run ends, the RAW camera
frames (no drawings; flipped if the page flips) are saved to runs/<this live demo>/run_<time>.mp4, plus
run_<time>.json: the dispenses with their time in the video, for annotating later. The window shows a REC mark.
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

from pipette_coach.config import (  # noqa: E402
    COMPARE_CV_MODEL,
    COMPARE_MODEL,
    LIQUID_MODEL,
    REPO_ROOT,
    TIP_MODEL,
    WELL_NET,
    load_config,
)
from pipette_coach.liquid_live import ReleaseCheck  # noqa: E402
from pipette_coach.live_pipeline import GPU_LOCK, LatestFrame, PlateWorker, TipFinder, serialize_gpu  # noqa: E402
from pipette_coach.localizer.detector import PlateDetector, _default_device  # noqa: E402
from pipette_coach.localizer.geometry import GRID, pitch_px, project  # noqa: E402
from pipette_coach.localizer.live import PlateTracker  # noqa: E402
from pipette_coach.localizer.well_net import WellNet  # noqa: E402
from pipette_coach.recorder import VideoRecorder  # noqa: E402
from pipette_coach.runs import new_run_dir, write_metrics  # noqa: E402
from pipette_coach.tips import DwellDetector  # noqa: E402
from pipette_coach.ui_link import UiLink  # noqa: E402
from pipette_coach.well_change import DispenseCheck  # noqa: E402
from pipette_coach.well_compare import CompareCheck  # noqa: E402


def well_name(w) -> str:
    return f"{'ABCDEFGH'[w[1]]}{w[0] + 1}"


def overlay(frame, plate, tip, well, dwell, t, events, fps_shown, liq=None, liq_thr=0.5):
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
            if liq is not None:  # the liquid model's verdict for this well now (green = liquid going in)
                lc = (0, 220, 0) if liq > liq_thr else (200, 200, 200)
                cv2.putText(img, f"liquid {liq:.2f}", (int(c[0]) + rad, int(c[1]) + rad + 24), 0, 0.8, lc, 2,
                            cv2.LINE_AA)  # fmt: skip
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
    ap.add_argument("--model", default=TIP_MODEL)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--show", action="store_true", help="show a window (q quits)")
    ap.add_argument("--save", action="store_true", help="also write the shown frames to live.mp4")
    ap.add_argument("--name", default="live-demo")
    ap.add_argument("--dump", default=None, help="write each frame's plate homography, tip and dwell to this .jsonl")
    ap.add_argument("--liquid", default="none",
                    choices=["model", "rule", "compare", "compare-frame", "compare-cv", "none"],
                    help="liquid check: model = one-well model while liquid goes in (pipette_coach/liquid_live.py), "
                         "rule = before/after rule after the tip leaves (well_change.py), compare = before/after "
                         "model on median views, compare-frame = on single frames 3 frames before / after "
                         "(pipette_coach/well_compare.py), compare-cv = the k-fold model (all pipette8 presses) on "
                         "median views, none (default) = dwell only: --dwell-s on one well = dispensed")
    ap.add_argument("--dwell-s", type=float, default=None,
                    help="seconds the tip must stay on a well to count as a dispense (default: 1.5 with --liquid none, "
                         "0.4 with a liquid check, which then decides)")
    ap.add_argument("--compare-model", default=COMPARE_MODEL)
    ap.add_argument("--liquid-model", default=LIQUID_MODEL)
    ap.add_argument("--liquid-thr", type=float, default=0.95, help="model: probability that counts as liquid")
    ap.add_argument("--no-well-change", action="store_true", help="same as --liquid none")
    ap.add_argument(
        "--rows-toward",
        choices=["down", "up", "left", "right"],
        default=None,
        help="image direction of rows A -> H (default: A1 = corner nearest the image top-left)",
    )
    ap.add_argument("--size", default="1920x1080", help="resolution to ask the camera for (WxH)")
    ap.add_argument("--ui", default=None, help="Well Plate UI address, e.g. http://localhost:7500")
    ap.add_argument(
        "--flip",
        choices=["none", "h", "v", "hv"],
        default=None,
        help="flip every frame: h left-right, v up-down (default: the page's flip buttons with --ui)",
    )
    a = ap.parse_args()
    if a.no_well_change:
        a.liquid = "none"
    if a.dwell_s is None:
        a.dwell_s = 1.5 if a.liquid == "none" else 0.4
    from ultralytics import YOLO

    cfg = load_config("localizer")
    cfg["detector"]["weights"] = "models/plate_det/weights/best.pt"
    dev = _default_device()
    det, net = PlateDetector(cfg), WellNet(REPO_ROOT / WELL_NET)
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
    cam.flip = {"none": None, "h": 1, "v": 0, "hv": -1}[a.flip] if a.flip else (ui.flip if ui else None)
    if a.camera is not None:
        cam.next(-1)  # the first frame tells the real size
        print(f"camera {a.camera}: {cam.size[0]}x{cam.size[1]} at {cam.fps:.0f} fps", flush=True)
    plate = PlateWorker(tracker, cam)
    check = DispenseCheck() if a.liquid == "rule" else None
    if a.liquid.startswith("compare"):
        weights = REPO_ROOT / (COMPARE_CV_MODEL if a.liquid == "compare-cv" else a.compare_model)
        if not weights.exists():
            ap.error(f"{weights} not found (compare-cv: python scripts/train_compare_kfold.py makes it)")
        check = CompareCheck(weights, "frame" if a.liquid == "compare-frame" else "median", dev, lock=GPU_LOCK)
    release = ReleaseCheck(a.liquid_model, dev, thr=a.liquid_thr, lock=GPU_LOCK) if a.liquid == "model" else None
    # Without a liquid check (default): every stop of dwell_s (1.5 s) on one well is a dispense; gaps up to 0.5 s (a
    # missed tip detection) do not restart the count. With a liquid check every stop >= 0.4 s is its own candidate
    # and counts only if liquid shows.
    if check or release:
        dwell = DwellDetector(min_s=a.dwell_s, trip_gap_s=None if check else 1.0)
    else:
        dwell = DwellDetector(min_s=a.dwell_s, gap_s=0.5, trip_gap_s=None)
    events = []
    run = new_run_dir(a.name)
    dump = open(a.dump, "w", buffering=1) if a.dump else None  # line-buffered: complete even if the run is cut
    # processing runs in its own thread; the main thread only shows the newest result (macOS allows windows on
    # the main thread only, and showing a frame costs ~15-20 ms - it must not hold up processing)
    st = {"img": None, "img_idx": -1, "shown": 0, "loop_ms": [], "grid_age": [], "t_first": None, "fps": 0.0,
          "stop": False, "vw": None, "why": None, "rec": None, "rec_run": None, "rec_t": 0.0,
          "recordings": []}  # fmt: skip

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
            if ui and not a.flip and cam.flip != ui.flip:  # flip changed on the page (the grid re-finds the plate)
                cam.flip = ui.flip
                print(f"{t:7.2f}s flip now {ui.flip} (from the Well Plate UI)", flush=True)
            if ui and not a.rows_toward and tracker.rows_toward != ui.rows_toward:  # A1 corner changed on the page
                tracker.rows_toward = ui.rows_toward
                print(f"{t:7.2f}s rows A -> H now {ui.rows_toward or 'auto'} (from the Well Plate UI)", flush=True)
            p = plate.latest()
            tp, well = tip(frame, p)
            if ui:
                ui.tip(well_name(well) if well is not None else None)  # live ring on the page (sent on change only)
            fired = release.update(t, frame, p.get("H"), well) if release else None
            if fired is not None:  # the liquid model saw liquid going into the well under the tip
                events.append({"t": round(t, 2), "well": well_name(fired), "liquid": True, "p": round(release.p, 2)})
                print(f"{t:7.2f}s dispense {well_name(fired)}: liquid (p={release.p:.2f})", flush=True)
                if ui:
                    ui.report(well_name(fired))
            ev = dwell.update(t, well)
            if ev is not None and not release:  # with the model, dispenses come from it (the dwell ring stays)
                if dwell.revised:  # a longer stop in the same trip: that was the dispense, not the pause before it
                    print(f"{t:7.2f}s dispense {events[-1]['well']} -> {well_name(ev)}", flush=True)
                    events[-1] = {**events[-1], "well": well_name(ev), "revised_from": events[-1]["well"]}
                else:
                    events.append({"t": round(t, 2), "well": well_name(ev)})
                    print(f"{t:7.2f}s dispense {well_name(ev)}", flush=True)
                if check:
                    check.watch(len(events) - 1, ev, dwell.start)
                elif ui:  # dwell only: the stop itself is the dispense
                    ui.report(well_name(ev))
            for idx, r in check.update(t, frame, p.get("H"), well) if check else []:  # liquid seen in the well?
                events[idx].update(liquid=r["filled"], after_s=r["after_s"],
                                   **({"p": round(r["p"], 3) if r["p"] is not None else None} if "p" in r
                                      else {"tex": round(r["tex"], 2), "de": round(r["de"], 1)}))  # fmt: skip
                print(f"{t:7.2f}s {events[idx]['well']}: liquid {r['filled']} ({r['after_s']:.2f}s after the tip left)",
                      flush=True)  # fmt: skip
                if ui and r["filled"] is True:
                    ui.report(events[idx]["well"])
            if dump is not None:
                H = p.get("H")
                dump.write(json.dumps({"t": round(t, 3), "H": None if H is None else np.asarray(H).tolist(),
                                       "tip": None if tp is None else np.asarray(tp, float).tolist(),
                                       "well": None if well is None else well_name(well),
                                       "dwell": round(float(dwell.progress(t)), 3)}) + "\n")  # fmt: skip
            img = overlay(frame, p, tp, well, dwell, t, events, st["fps"], release.p if release else None, a.liquid_thr)
            if a.save:
                if st["vw"] is None:
                    st["vw"] = cv2.VideoWriter(str(run / "live.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cam.fps,
                                               img.shape[1::-1])  # fmt: skip
                st["vw"].write(img)
            record(frame, img, t, t0)
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

    def finish_recording(background: bool) -> None:
        """Close the open recording and write its dispenses (time in the video) next to it."""
        rec, t_start = st["rec"], st["rec_t"]
        done = [{**e, "t_video": round(e["t"] - t_start, 2)} for e in events if e["t"] >= t_start]
        side = {"video": rec.path.name, "fps": rec.fps, "camera_t_start": round(t_start, 2), "dispenses": done}
        rec.path.with_suffix(".json").write_text(json.dumps(side, indent=1))
        if background:
            threading.Thread(target=rec.close, daemon=True).start()
        else:
            rec.close()
        print(f"recording saved: {rec.path} ({len(side['dispenses'])} dispenses)", flush=True)
        st["rec"] = None

    def record(frame: np.ndarray, img: np.ndarray, t: float, now: float) -> None:
        """The page's "Record each run": the raw frame goes to a new file when a run starts, closed when it ends
        (closing in the background, so the loop never waits for the encoder). REC is drawn on the window only."""
        want = ui.record_run if ui else None
        if want != st["rec_run"]:
            if st["rec"] is not None:
                finish_recording(background=True)
            st["rec_run"] = want
            if want is not None:
                st["rec"] = VideoRecorder(run / f"run_{time.strftime('%Y%m%d-%H%M%S')}.mp4", fps=cam.fps or 30.0)
                st["rec_t"] = t
                st["recordings"].append(str(st["rec"].path))
                print(f"recording (raw): {st['rec'].path}", flush=True)
            if ui:
                ui.recording = st["rec"].path.name if st["rec"] else None
        if st["rec"] is not None:
            st["rec"].write(frame, now)
            cv2.circle(img, (img.shape[1] - 150, 40), 13, (40, 40, 230), -1)
            cv2.putText(img, "REC", (img.shape[1] - 128, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 230), 3)

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
        if st["rec"] is not None:
            finish_recording(background=False)
        cv2.destroyAllWindows()
    shown, loop_ms, grid_age = st["shown"], st["loop_ms"], st["grid_age"]
    lm = np.array(loop_ms) if loop_ms else np.zeros(1)
    m = {"source": str(src), "start": a.start, "model": a.model, "imgsz": a.imgsz, "device": dev, "liquid": a.liquid,
         "liquid_model": a.liquid_model if a.liquid == "model" else None,
         "camera_fps": cam.fps, "seconds": round(elapsed, 1), "camera_frames": cam.read_count,
         "processed_frames": shown,
         "processed_fps": shown / max(elapsed, 1e-6), "processed_share": shown / max(cam.read_count, 1),
         "loop_ms_p50": float(np.median(lm)), "loop_ms_p95": float(np.percentile(lm, 95)),
         "plate_updates_per_s": plate.updates / max(elapsed, 1e-6),
         "grid_age_frames_p50": float(np.median(grid_age)) if grid_age else None,
         "grid_age_frames_p95": float(np.percentile(grid_age, 95)) if grid_age else None,
         "window_frames_per_s": displayed / max(elapsed, 1e-6) if a.show else None,
         "stopped_because": st["why"], "dispenses": events, "saved_video": bool(a.save), "recordings": st["recordings"],
         "ui": None if ui is None else {"url": a.ui, "sent": ui.sent, "errors": ui.errors[-20:]}}  # fmt: skip
    write_metrics(run, m)
    (run / "plate_log.json").write_text(json.dumps(plate.log))
    if release:
        (run / "liquid_trace.json").write_text(json.dumps(release.trace))
    (run / "notes.md").write_text(f"# Live demo ({src})\n\n```\n{json.dumps(m, indent=1)}\n```\n\nKeep local.\n")
    print(json.dumps({k: v for k, v in m.items() if k != "dispenses"}, indent=1), "\n->", run)


if __name__ == "__main__":
    main()
