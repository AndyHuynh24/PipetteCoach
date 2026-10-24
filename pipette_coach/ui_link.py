"""Link to the Well Plate UI (wellPlateUiWeb/): send confirmed dispenses, read where A1 is.

All HTTP runs in one background thread, so a slow or missing server never holds up the camera loop: report() only
puts the well on a queue. tip() tells the page where the tip is (live green / red ring): only changes are sent, the
newest one wins, and a tip lost for less than `tip_hold_s` (detector flicker) is not sent as "no tip".
Every `heartbeat_s` the thread also sends a heartbeat (the page shows "Camera: online") and
re-reads the plan's A1 corner, which the page lets the user set ("A1 in the camera picture").
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


def rows_toward_for(corner: str | None) -> str | None:
    """None for "auto" (the tracker's own guess)."""
    return CORNER_ROWS.get(corner or "")


class UiLink:
    def __init__(self, url: str, token: str | None = None, heartbeat_s: float = 3.0, timeout_s: float = 2.0,
                 tip_hold_s: float = 0.3):  # fmt: skip
        self.url, self.token, self.heartbeat_s, self.timeout_s = url.rstrip("/"), token, heartbeat_s, timeout_s
        self.tip_hold_s = tip_hold_s
        self._tip, self._tip_sent, self._tip_queued, self._tip_lost = None, None, False, None
        self.rows_toward: str | None = None
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
        """Heartbeat + the plan's A1 corner. Errors are kept, never raised."""
        try:
            self._call("POST", "/api/camera/heartbeat", {})
            self.rows_toward = rows_toward_for(self._call("GET", "/api/plan").get("a1Corner"))
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
            try:
                well = self._q.get(timeout=self.heartbeat_s)
            except queue.Empty:
                self.refresh()
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
