import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pipette_coach.ui_link import UiLink, rows_toward_for


class FakeUi(BaseHTTPRequestHandler):
    """Stands in for wellPlateUiWeb: records requests, answers the plan with a set A1 corner."""

    calls: list = []
    corner = "bottomLeft"

    def _answer(self, body: dict) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        FakeUi.calls.append(("GET", self.path, None, self.headers.get("X-Api-Token")))
        self._answer({"a1Corner": FakeUi.corner})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        FakeUi.calls.append(("POST", self.path, body, self.headers.get("X-Api-Token")))
        if self.path == "/api/drop":
            time.sleep(0.3)  # a slow server must not slow report()
        self._answer({"accepted": True})

    def log_message(self, *args):
        pass


def test_rows_toward_for_each_corner():
    assert [rows_toward_for(c) for c in ("topLeft", "bottomRight", "topRight", "bottomLeft")] == [
        "down",
        "up",
        "left",
        "right",
    ]
    assert rows_toward_for("auto") is None and rows_toward_for(None) is None


def test_reports_reach_the_server_without_blocking_and_corner_is_read():
    FakeUi.calls.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeUi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        link = UiLink(f"http://127.0.0.1:{server.server_address[1]}", token="tok").start()
        assert link.rows_toward == "right"  # bottomLeft, read once before the camera starts
        t0 = time.perf_counter()
        link.report("A1")
        link.report("B1")
        assert time.perf_counter() - t0 < 0.05
        link.stop()
        drops = [c[2]["wells"] for c in FakeUi.calls if c[1] == "/api/drop"]
        assert drops == [["A1"], ["B1"]]
        assert all(c[3] == "tok" for c in FakeUi.calls)
        assert [w for w, _ in link.sent] == ["A1", "B1"] and link.errors == []
    finally:
        server.shutdown()


def test_missing_server_is_not_an_error_for_the_caller():
    link = UiLink("http://127.0.0.1:9", timeout_s=0.2).start()  # nothing listens on port 9
    assert link.rows_toward is None and link.errors
    link.report("C3")  # still returns at once
    link.stop()
    assert any("drop C3" in e for e in link.errors)


def test_tip_sends_changes_only_and_ignores_short_dropouts():
    FakeUi.calls.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeUi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        link = UiLink(f"http://127.0.0.1:{server.server_address[1]}", tip_hold_s=0.2).start()
        for w in ["B4"] * 5 + [None] + ["B4"] * 3:  # one dropped frame: still B4
            link.tip(w)
        time.sleep(0.2)
        link.tip("C4")
        time.sleep(0.2)
        link.tip(None)  # lost: not sent yet
        time.sleep(0.25)
        link.tip(None)  # lost for longer than tip_hold_s: sent
        time.sleep(0.2)
        link.stop()
        tips = [c[2]["well"] for c in FakeUi.calls if c[1] == "/api/camera/tip"]
        assert tips == ["B4", "C4", None] and link.errors == []
    finally:
        server.shutdown()
