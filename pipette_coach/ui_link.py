"""Link to the Well Plate UI (wellPlateUiWeb/): send confirmed dispenses, read where A1 is.

All HTTP runs in one background thread, so a slow or missing server never holds up the camera loop: report() only
puts the well on a queue. tip() tells the page where the tip is (live green / red ring): only changes are sent, the
newest one wins, and a tip lost for less than `tip_hold_s` (detector flicker) is not sent as "no tip".
Every `heartbeat_s` (also while tips and drops keep coming) the thread sends a heartbeat (the page shows "Camera:
online", and "REC" while `recording` is set) and re-reads the plan: the A1 corner and flip the page lets the user set,
and whether this run should be recorded (`record_run`: the run's start time while "Record each run" is on and the run
is going, else None).
Standard library only.
"""

import json
import queue
import threading
import time
import urllib.request

# A1 corner in the picture -> image direction of rows A -> H (PlateTracker rows_toward). Each corner fits one plate
# orientation: landscape plate A1 top-left / bottom-right, portrait plate A1 top-right / bottom-left.
CORNER_ROWS = {"topLeft": "down", "bottomRight": "up", "topRight": "left", "bottomLeft": "right"}
_TIP = object()  # queue marker: send the newest tip well


def flip_code_for(flip: dict | None) -> int | None:
    """The page's {"flipH", "flipV"} -> cv2.flip code (1 left-right, 0 up-down, -1 both) or None (no flip)."""
    h, v = bool((flip or {}).get("flipH")), bool((flip or {}).get("flipV"))
    return -1 if h and v else 1 if h else 0 if v else None


def rows_toward_for(corner: str | None) -> str | None:
    """None for "auto" (the tracker's own guess)."""
    return CORNER_ROWS.get(corner or "")


class UiLink:
    def __init__(self, url: str, token: str | None = None, heartbeat_s: float = 1.0, timeout_s: float = 2.0,
                 tip_hold_s: float = 0.3):  # fmt: skip
        self.url, self.token, self.heartbeat_s, self.timeout_s = url.rstrip("/"), token, heartbeat_s, timeout_s
        self.tip_hold_s = tip_hold_s
        self._tip, self._tip_sent, self._tip_queued, self._tip_lost = None, None, False, None
        self.rows_toward: str | None = None
        self.flip: int | None = None  # cv2.flip code from the page's flip buttons
        self.record_run: float | None = None  # start time of the run to record (None: do not record)
        self.recording: str | None = None  # set by the caller: the file being recorded, sent with each heartbeat
        self._refreshed = 0.0
        self.sent: list[tuple[str, dict]] = []  # (well, server verdict)
        self.errors: list[str] = []
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Api-Token"] = self.token
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read() or b"{}")

    def refresh(self) -> None:
        """Heartbeat + the plan's A1 corner and camera flip. Errors are kept, never raised."""
        self._refreshed = time.monotonic()
        try:
            self._call("POST", "/api/camera/heartbeat", {"recording": self.recording})
            plan = self._call("GET", "/api/plan")
            self.rows_toward = rows_toward_for(plan.get("a1Corner"))
            self.flip = flip_code_for(plan.get("cameraFlip"))
            go = plan.get("record") and plan.get("status") == "active"
            self.record_run = plan.get("startedAt") if go else None
        except Exception as e:  # server not up yet / restarting: keep going, try again next time
            self.errors.append(f"refresh: {e}")

    def start(self) -> "UiLink":
        self.refresh()  # once before the camera starts, so the first grid already uses the corner
        self._thread.start()
        return self

    def report(self, well: str) -> None:
        """A dispense with liquid seen in `well` (e.g. "B4"). Returns at once."""
        self._q.put(well)

    def tip(self, well: str | None) -> None:
        """The well the tip is over in this frame (None = no tip on a well). Returns at once."""
        if well is None and self._tip is not None:  # lost: wait tip_hold_s before saying so
            now = time.monotonic()
            self._tip_lost = self._tip_lost or now
            if now - self._tip_lost < self.tip_hold_s:
                return
        self._tip_lost = None
        if well != self._tip:
            self._tip = well
            if not self._tip_queued:
                self._tip_queued = True
                self._q.put(_TIP)

    def _run(self) -> None:
        while not self._stop.is_set():
            if time.monotonic() - self._refreshed >= self.heartbeat_s:
                self.refresh()  # due even when tips keep the queue busy
            try:
                well = self._q.get(timeout=self.heartbeat_s)
            except queue.Empty:
                continue
            if well is None:  # stop() was called and everything before it is sent
                return
            if well is _TIP:
                self._tip_queued = False  # before reading: a change from now on queues a new send
                tip = self._tip
                if tip != self._tip_sent:
                    try:
                        self._call("POST", "/api/camera/tip", {"well": tip})
                        self._tip_sent = tip
                    except Exception as e:
                        self.errors.append(f"tip {tip}: {e}")
                continue
            try:
                res = self._call("POST", "/api/drop", {"wells": [well]})
                self.sent.append((well, {k: res.get(k) for k in ("accepted", "reason", "results", "alarm")}))
            except Exception as e:
                self.errors.append(f"drop {well}: {e}")

    def stop(self, wait_s: float = 3.0) -> None:
        """Send what is still queued (waits up to `wait_s`), then stop."""
        self._q.put(None)
        self._thread.join(timeout=wait_s)
        self._stop.set()
