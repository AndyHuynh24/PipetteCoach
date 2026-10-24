"""Well Plate UI: web server.

    pip install -r requirements.txt
    python app.py                     # then open http://localhost:7500

The browser shows the UI. The camera code (scripts/live_demo.py --ui) reports
every dispense it has confirmed with one HTTP call:

    POST /api/drop   {"wells": ["B4"]}

The server compares it with the plan, and if the well is wrong it raises an
alarm that every open browser shows immediately (Server-Sent Events).

Only two third-party packages are needed: starlette and uvicorn.
"""

import argparse
import asyncio
import contextlib
import functools
import hmac
import json
import logging
import os
import socket
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `python wellPlateUiWeb/app.py` works from any folder

import plateLogic as pl  # noqa: E402
import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.exceptions import HTTPException  # noqa: E402
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402
from store import PlateStore, nowIso  # noqa: E402

DEFAULT_DB = os.path.join(HERE, "data", "wellPlate.db")
STATIC_DIR = os.path.join(HERE, "static")
CAMERA_ONLINE_SECONDS = 8  # no heartbeat or drop for this long = camera shown as offline

log = logging.getLogger("wellplate")


class FreshStatic(StaticFiles):
    """Static files the browser re-checks on every load (a cheap 304 when unchanged), so a reload after an update
    never runs an old app.js against a new server."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


class Hub:
    """Fan-out of server events to every open browser tab."""

    def __init__(self):
        self.clients = set()

    def subscribe(self):
        q = asyncio.Queue(maxsize=100)
        self.clients.add(q)
        return q

    def unsubscribe(self, q):
        self.clients.discard(q)

    def publish(self, eventName, data):
        message = f"event: {eventName}\ndata: {json.dumps(data)}\n\n"
        for q in list(self.clients):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # A tab that stopped reading. Cut it loose; the browser
                # reconnects by itself and receives a fresh full state.
                while not q.empty():
                    q.get_nowait()
                q.put_nowait(None)
                self.clients.discard(q)


class CameraLink:
    """Tracks whether the camera code is alive (it calls /api/camera/heartbeat)."""

    def __init__(self):
        self.lastSeenMono = None
        self.lastSeenIso = None

    def touch(self):
        self.lastSeenMono = time.monotonic()
        self.lastSeenIso = nowIso()

    def online(self):
        return self.lastSeenMono is not None and time.monotonic() - self.lastSeenMono < CAMERA_ONLINE_SECONDS

    def snapshot(self):
        return {"online": self.online(), "lastSeen": self.lastSeenIso}


def createApp(dbPath=DEFAULT_DB, apiToken=None, seedDemo=True):
    store = PlateStore(dbPath, seedDemo=seedDemo)
    hub = Hub()
    camera = CameraLink()

    # Where the camera (or the simulator page) sees the tip right now. Memory only: it changes many times a second.
    tip = {"well": None, "since": None, "added": False}
    serverId = uuid.uuid4().hex  # changes on every restart
    revision = [0]

    def fullState():
        """Every state carries a number that only goes up. A page that gets an
        old answer after a newer live update can tell, and ignores the old one."""
        revision[0] += 1
        state = store.getState()
        state["camera"] = camera.snapshot()
        state["tip"] = tipInfo()
        state["serverId"] = serverId
        state["revision"] = revision[0]
        return state

    def pushState():
        hub.publish("state", fullState())

    def tipInfo():
        """The tip's well and what a drop there would be: ok (green on the page) or a mistake kind (red).
        "added" = the tip is still over the well it just filled correctly (not a duplicate)."""
        if tip["well"] is None:
            return {"well": None}
        verdict = store.judgeTip(tip["well"]) or {"kind": None}
        if tip["added"] and verdict["kind"] == "duplicate":
            verdict["kind"] = "added"
        return {"well": tip["well"], "since": tip["since"], **verdict}

    def setTip(well):
        well = None if well in (None, "") else pl.normalizeWells([well])[0]
        if well != tip["well"]:
            tip.update(well=well, since=time.time(), added=False)
            hub.publish("tip", tipInfo())

    def reply(payload=None, status=200, publish=True):
        """Answer a change with the new full state and push the same state to
        every other open page, so two screens never disagree."""
        state = fullState()
        if publish:
            hub.publish("state", state)
        body = dict(payload or {})
        body["state"] = state
        return JSONResponse(body, status_code=status)

    def api(fn):
        """Turn the exceptions the logic raises into clean JSON errors."""

        @functools.wraps(fn)
        async def wrapper(request):
            try:
                return await fn(request)
            except pl.NotFoundError as e:
                return JSONResponse({"error": str(e)}, status_code=404)
            except PermissionError as e:
                return JSONResponse({"error": str(e)}, status_code=401)
            except ValueError as e:  # includes bad JSON
                return JSONResponse({"error": str(e)}, status_code=400)

        return wrapper

    async def readJson(request):
        try:
            data = await request.json()
        except Exception:
            raise ValueError("The request body must be valid JSON.") from None
        if not isinstance(data, dict):
            raise ValueError("The request body must be a JSON object.")
        return data

    def cameraCall(request):
        """The camera endpoints need the token (if one is set) and mark the camera online."""
        if apiToken and not hmac.compare_digest(request.headers.get("x-api-token", ""), apiToken):
            raise PermissionError("Missing or wrong X-Api-Token header.")
        cameraSeen()

    def cameraSeen():
        wasOnline = camera.online()
        camera.touch()
        if not wasOnline:
            pushState()

    # ------------------------------------------------------------ pages and reading
    async def index(request):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers={"Cache-Control": "no-cache"})

    async def simulatorPage(request):
        return FileResponse(os.path.join(STATIC_DIR, "camera.html"), headers={"Cache-Control": "no-cache"})

    async def health(request):
        return JSONResponse({"ok": True, "time": nowIso(), "tokenRequired": bool(apiToken)})

    @api
    async def getState(request):
        return JSONResponse(fullState())

    @api
    async def getPlan(request):
        return JSONResponse(store.getPlanPayload())

    async def stream(request):
        q = hub.subscribe()

        async def events():
            try:
                yield "retry: 2000\n\n"
                yield f"event: state\ndata: {json.dumps(fullState())}\n\n"
                while True:
                    try:
                        message = await asyncio.wait_for(q.get(), timeout=15)
                    except TimeoutError:
                        yield ": ping\n\n"  # keeps proxies and the browser from giving up
                        continue
                    if message is None:
                        return
                    yield message
            finally:
                hub.unsubscribe(q)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # ------------------------------------------------------------ compounds
    @api
    async def createCompound(request):
        return reply({"compound": store.saveCompound(await readJson(request))}, 201)

    @api
    async def updateCompound(request):
        return reply({"compound": store.saveCompound(await readJson(request), request.path_params["compoundId"])})

    @api
    async def deleteCompound(request):
        return reply({"deleted": store.deleteCompound(request.path_params["compoundId"])})

    # ------------------------------------------------------------ plan and run
    @api
    async def putPlan(request):
        return reply(store.savePlan((await readJson(request)).get("plan", [])))

    @api
    async def setActiveCompound(request):
        store.setActiveCompound(pl.optionalId((await readJson(request)).get("compoundId"), "compoundId"))
        return reply()

    @api
    async def setA1Corner(request):
        store.setA1Corner((await readJson(request)).get("a1Corner"))
        return reply()

    @api
    async def startRun(request):
        store.startRun()
        return reply()

    @api
    async def resetRun(request):
        store.resetRun()
        return reply()

    @api
    async def nextWell(request):
        return reply(store.markWellDone((await readJson(request)).get("well", "")))

    @api
    async def acknowledgeAlarm(request):
        store.acknowledgeAlarm()
        return reply()

    # ------------------------------------------------------------ drops
    async def handleDrop(request, source):
        data = await readJson(request)
        compoundId = pl.optionalId(data.get("compoundId"), "compoundId")
        result = store.reportDrop(data.get("wells", data.get("well")), compoundId, source)
        if result["accepted"]:
            for r in result["results"]:
                if r["kind"] == "ok" and r["well"] == tip["well"]:
                    tip["added"] = True
                log.info("%s drop %s -> %s (planned: %s)", source, r["well"], r["kind"], r["planned"])
        return reply(result, publish=result["accepted"])  # a refused drop changed nothing

    @api
    async def postDrop(request):
        """The camera code calls this once per confirmed dispense."""
        cameraCall(request)
        return await handleDrop(request, "camera")

    @api
    async def postTestDrop(request):
        """The 'Test' buttons and well taps in the Run tab call this."""
        return await handleDrop(request, "test")

    @api
    async def postSimDrop(request):
        """The camera simulator page (/camera) dispenses: judged exactly like a camera drop. Open like the
        test buttons (no token), since it is a page on this server."""
        cameraSeen()
        return await handleDrop(request, "simulator")

    @api
    async def postCameraTip(request):
        """The camera code: the well the tip is over now ({"well": null} = no tip). Sent when it changes."""
        cameraCall(request)
        setTip((await readJson(request)).get("well"))
        return JSONResponse({"ok": True})

    @api
    async def postSimTip(request):
        cameraSeen()
        setTip((await readJson(request)).get("well"))
        return JSONResponse({"ok": True})

    @api
    async def postSimHeartbeat(request):
        cameraSeen()
        return JSONResponse({"ok": True})

    @api
    async def postHeartbeat(request):
        """The camera code calls this every few seconds so the page can show 'Camera: online'."""
        cameraCall(request)
        return JSONResponse({"ok": True})

    # ------------------------------------------------------------ lifecycle
    @contextlib.asynccontextmanager
    async def lifespan(app):
        async def watchCamera():
            last = camera.online()
            while True:
                await asyncio.sleep(2)
                if camera.online() != last:
                    last = camera.online()
                    if not last:
                        tip.update(well=None, since=None, added=False)  # no camera, no tip
                    pushState()  # camera went quiet: tell every open page

        task = asyncio.create_task(watchCamera())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    routes = [
        Route("/", index),
        Route("/camera", simulatorPage),
        Route("/api/health", health),
        Route("/api/state", getState),
        Route("/api/plan", getPlan, methods=["GET"]),
        Route("/api/plan", putPlan, methods=["PUT"]),
        Route("/api/stream", stream),
        Route("/api/compounds", createCompound, methods=["POST"]),
        Route("/api/compounds/{compoundId:int}", updateCompound, methods=["PUT"]),
        Route("/api/compounds/{compoundId:int}", deleteCompound, methods=["DELETE"]),
        Route("/api/activeCompound", setActiveCompound, methods=["POST"]),
        Route("/api/a1Corner", setA1Corner, methods=["POST"]),
        Route("/api/run/start", startRun, methods=["POST"]),
        Route("/api/run/reset", resetRun, methods=["POST"]),
        Route("/api/run/nextWell", nextWell, methods=["POST"]),
        Route("/api/alarm/ack", acknowledgeAlarm, methods=["POST"]),
        Route("/api/drop", postDrop, methods=["POST"]),
        Route("/api/testDrop", postTestDrop, methods=["POST"]),
        Route("/api/camera/heartbeat", postHeartbeat, methods=["POST"]),
        Route("/api/camera/tip", postCameraTip, methods=["POST"]),
        Route("/api/sim/drop", postSimDrop, methods=["POST"]),
        Route("/api/sim/tip", postSimTip, methods=["POST"]),
        Route("/api/sim/heartbeat", postSimHeartbeat, methods=["POST"]),
        Mount("/static", FreshStatic(directory=STATIC_DIR), name="static"),
    ]

    async def httpError(request, exc):
        """Unknown address or wrong method: answer /api calls with JSON like every other error."""
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": exc.detail or "Request failed."}, status_code=exc.status_code)
        return PlainTextResponse(exc.detail or "Request failed.", status_code=exc.status_code)

    app = Starlette(routes=routes, lifespan=lifespan, exception_handlers={HTTPException: httpError})
    app.state.store = store
    app.state.hub = hub
    app.state.camera = camera
    return app


def lanAddress():
    """Best guess at this computer's address on the local network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; it only picks a route
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def main():
    parser = argparse.ArgumentParser(description="Well Plate UI server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="127.0.0.1 = this computer only (default). 0.0.0.0 = other devices on the network too.",
    )
    parser.add_argument("--port", type=int, default=7500)
    parser.add_argument("--db", default=DEFAULT_DB, help="path of the SQLite file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    token = os.environ.get("WELLPLATE_API_TOKEN") or None
    app = createApp(args.db, token)

    print("\nWell Plate UI")
    print(f"  Open on this computer:  http://localhost:{args.port}")
    if args.host != "127.0.0.1" and (ip := lanAddress()):
        print(f"  Open from a tablet:     http://{ip}:{args.port}")
    tokenNote = (
        "needs the X-Api-Token header" if token else "no token set: anyone who can reach this port can send drops"
    )
    print(f"  Camera endpoint:        POST /api/drop   ({tokenNote})")
    print(f"  Database:               {args.db}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
