"""Tests for Well Plate UI.

    python tests/runTests.py

Three layers:
  1. LogicTests   plateLogic.py on its own (no database, no network)
  2. StoreTests   store.py with an in-memory SQLite database
  3. ServerTests  the real web server (uvicorn) on a free port, called over HTTP,
                  including the live event stream and the fake camera script

Only the standard library plus starlette and uvicorn are needed.
"""

import http.client
import json
import os
import random
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import plateLogic as pl  # noqa: E402
from store import PlateStore  # noqa: E402


def S(wellMap):
    """{"A1": 1, "B1": 2} -> one add step per compound (the plan before a well could hold several)."""
    return pl.stepsFromWellMap(wellMap, list(dict.fromkeys(wellMap.values())))


def add(cid, *wells):
    return {"kind": "add", "compoundId": cid, "wells": list(wells)}


def wait(seconds):
    return {"kind": "wait", "seconds": seconds}


# ======================================================================== 1. logic
class LogicTests(unittest.TestCase):
    def testWellParsingAndNormalizing(self):
        self.assertEqual(pl.parseWell("A1"), (0, 0))
        self.assertEqual(pl.parseWell(" b04 "), (1, 3))
        self.assertEqual(pl.normalizeWell("h09"), "H9")
        for bad in ("", "4B", "AA1", None):
            with self.assertRaises(ValueError):
                pl.parseWell(bad)

    def testInPlateIs96WellsOnly(self):
        self.assertTrue(pl.inPlate("H12"))
        self.assertFalse(pl.inPlate("I1"))
        self.assertFalse(pl.inPlate("A13"))
        self.assertFalse(pl.inPlate("junk"))

    def testNormalizeWells(self):
        self.assertEqual(pl.normalizeWells("b4"), ["B4"])
        self.assertEqual(pl.normalizeWells(["A1", "a1", "B2"]), ["A1", "B2"])
        for bad in ([], None, ["A13"], ["Z9"]):
            with self.assertRaises(ValueError):
                pl.normalizeWells(bad)

    def testA1Corner(self):
        self.assertEqual(pl.normalizeA1Corner("bottomLeft"), "bottomLeft")
        with self.assertRaises(ValueError):
            pl.normalizeA1Corner("middle")

    def testOptionalId(self):
        self.assertIsNone(pl.optionalId(None))
        self.assertEqual(pl.optionalId("3"), 3)
        for bad in (True, "x", [1]):
            with self.assertRaises(ValueError):
                pl.optionalId(bad)

    def testValidateCompound(self):
        c = pl.validateCompound({"name": " Water ", "volumeUl": "5", "color": "#aabbcc"})
        self.assertEqual(c, {"name": "Water", "tag": "WAT", "volumeUl": 5.0, "color": "#AABBCC"})
        ok = {"name": "x", "volumeUl": 1, "color": "#000000"}
        for change in ({"name": ""}, {"volumeUl": "a"}, {"volumeUl": -1}, {"color": "red"}):
            with self.assertRaises(ValueError):
                pl.validateCompound({**ok, **change})

    def testCleanSteps(self):
        steps = [add(1, "b1", "a1", "A13", "a1"), add(9, "C1"), add(2), {"kind": "add", "compoundId": True}, wait("60")]
        self.assertEqual(pl.cleanSteps(steps, {1, 2}), [add(1, "A1", "B1"), wait(60)])
        for bad in ({"A1": 1}, [wait(0)], [wait(90000)], [wait("x")]):
            with self.assertRaises(ValueError):
                pl.cleanSteps(bad, {1})

    def testStepsFromWellMapAndSequences(self):
        steps = pl.stepsFromWellMap({"B1": 2, "A1": 1, "A2": 2}, [1, 2])
        self.assertEqual(steps, [add(1, "A1"), add(2, "A2", "B1")])
        self.assertEqual(pl.wellSequences([add(1, "A1"), wait(5), add(2, "A1", "B1")]), {"A1": [0, 2], "B1": [2]})

    def testClassifyKinds(self):
        steps = [add(1, "A1"), add(2, "A2")]
        res, done = pl.classifyDrops(["A1", "A2", "B1"], 1, steps, {}, {})
        self.assertEqual([r["kind"] for r in res], ["ok", "wrong", "unplanned"])
        self.assertEqual(done, {"0": {"A1": True}})  # a wrong drop does not mark the well done
        res, _ = pl.classifyDrops(["A1"], 1, steps, done, {})
        self.assertEqual(res[0]["kind"], "duplicate")

    def testSeveralCompoundsInOneWellGoInOrder(self):
        steps = [add(1, "A1"), add(2, "A1"), add(1, "A1")]  # e.g. sample, reagent, sample again
        res, done = pl.classifyDrops(["A1"], 2, steps, {}, {})
        self.assertEqual((res[0]["kind"], res[0]["plannedId"]), ("order", 1))  # reagent before the sample
        for cid in (1, 2, 1):
            res, done = pl.classifyDrops(["A1"], cid, steps, done, {})
            self.assertEqual(res[0]["kind"], "ok")
        self.assertTrue(pl.isComplete(steps, done))
        self.assertEqual(pl.classifyDrops(["A1"], 2, steps, done, {})[0][0]["kind"], "duplicate")

    def testTimersRunPerWell(self):
        steps = [add(1, "A1", "B1"), wait(60), add(2, "A1", "B1")]
        self.assertEqual(
            pl.wellTimers(steps, {}, 120.0)["A1"], [{"step": 1, "seconds": 60, "state": "pending", "endsAt": None}]
        )
        done = {"0": {"A1": 100.0}}  # A went into A1 at t=100; B1 not yet
        t = pl.wellTimers(steps, done, 130.0)
        self.assertEqual(
            (t["A1"][0]["state"], t["A1"][0]["endsAt"], t["B1"][0]["state"]), ("running", 160.0, "pending")
        )
        self.assertEqual(pl.classifyDrops(["A1"], 2, steps, done, t)[0][0]["kind"], "early")
        later = pl.wellTimers(steps, done, 160.0)
        self.assertEqual(pl.classifyDrops(["A1"], 2, steps, done, later, 160.0)[0][0]["kind"], "ok")  # B1 not needed
        res, new = pl.classifyDrops(["B1"], 1, steps, done, later, 170.0)
        self.assertEqual((res[0]["kind"], new["0"]["B1"]), ("ok", 170.0))  # the drop time starts B1's own timer
        self.assertEqual(pl.wellTimers(steps, new, 170.0)["B1"][0]["endsAt"], 230.0)

    def testTimerOnlyHoldsWellsWithACompoundBeforeAndAfterIt(self):
        steps = [wait(10), add(1, "A1"), wait(30), wait(20), add(2, "A1", "B1"), wait(5)]
        t = pl.wellTimers(steps, {"1": {"A1": 100.0}}, 0.0)
        self.assertEqual([(x["step"], x["endsAt"]) for x in t["A1"]], [(2, 130.0), (3, 150.0)])  # back to back
        self.assertNotIn("B1", t)  # B1's first compound comes after the timers
        self.assertEqual(pl.nextStepFor(steps, {"1": {"A1": 100.0}}, "A1"), 4)
        self.assertIsNone(pl.nextStepFor(steps, {}, "H12"))
        self.assertIsNotNone(pl.doneTime({"0": {"A1": True}}, 0, "A1"))  # old runs stored True

    def testProgressCountsEachCompoundInEachWell(self):
        steps = [add(1, "A1", "B1"), wait(5), add(2, "A1")]
        self.assertEqual(pl.progress(steps, {"0": {"A1": True}}), (1, 3))

    def testAlarmMessagesAndMerge(self):
        self.assertIn("needs S2 next", pl.alarmMessage("wrong", "A2", "S2", "S1"))
        self.assertIn("before the timer", pl.alarmMessage("early", "A2", "S2", "S1"))
        self.assertIn("needs S2 before S1", pl.alarmMessage("order", "A2", "S2", "S1"))
        self.assertIn("no compound planned", pl.alarmMessage("unplanned", "B1", "?", "S1"))
        self.assertIn("second drop", pl.alarmMessage("duplicate", "A1", "?", "S1"))
        a = pl.mergeAlarm(None, [{"well": "B2", "kind": "wrong"}], "m1", "t1")
        a = pl.mergeAlarm(a, [{"well": "A1", "kind": "unplanned"}], "m2", "t2")
        self.assertEqual(
            (a["kind"], a["well"], a["wells"], a["message"], a["count"]), ("wrong", "B2", ["A1", "B2"], "m1", 2)
        )

    def testIsComplete(self):
        self.assertFalse(pl.isComplete([], {}))
        self.assertFalse(pl.isComplete([wait(5)], {}))
        self.assertTrue(pl.isComplete([add(1, "A1")], {"0": {"A1": True}}))


# ======================================================================== 2. store
class StoreTests(unittest.TestCase):
    def setUp(self):
        self.s = PlateStore(":memory:")
        self.ids = [c["id"] for c in self.s.listCompounds()]  # the demo compounds
        self.a, self.b = self.ids[0], self.ids[1]

    def tearDown(self):
        self.s.conn.close()

    def startRun(self, plan, active=None):
        self.s.savePlan(S(plan))
        self.s.startRun()
        if active is not None:
            self.s.setActiveCompound(active)

    def testSeedsDemoOnceOnly(self):
        self.assertEqual(len(self.ids), len(pl.DEMO_COMPOUNDS))
        for cid in self.ids:
            self.s.deleteCompound(cid)
        self.assertEqual(self.s.listCompounds(), [])

    def testNoSeedWhenAsked(self):
        s = PlateStore(":memory:", seedDemo=False)
        self.assertEqual(s.listCompounds(), [])
        s.conn.close()

    def testSaveCompoundCreateUpdateAndErrors(self):
        c = self.s.saveCompound({"name": "Water", "volumeUl": 1, "color": "#000000"})
        self.assertEqual(c["position"], len(pl.DEMO_COMPOUNDS))
        self.assertEqual(self.s.saveCompound({"name": "W2", "volumeUl": 2, "color": "#000000"}, c["id"])["name"], "W2")
        with self.assertRaises(pl.NotFoundError):
            self.s.saveCompound({"name": "x", "volumeUl": 1, "color": "#000000"}, 999)

    def testDeleteRemovesWellsActiveAndResetsRun(self):
        self.startRun({"A1": self.a, "B1": self.b}, active=self.a)
        res = self.s.deleteCompound(self.a)
        self.assertEqual(res, {"wells": 1, "runReset": True})
        plate = self.s.getPlate()
        self.assertEqual(
            (plate["plan"], plate["status"], plate["activeCompoundId"]), ([add(self.b, "B1")], "draft", None)
        )
        with self.assertRaises(pl.NotFoundError):
            self.s.deleteCompound(self.a)

    def testEditingWellsResetsRun(self):
        self.startRun({"A1": self.a})
        self.assertEqual(self.s.savePlan(S({"A1": self.a}))["runReset"], False)  # same plan: run continues
        self.assertEqual(self.s.savePlan(S({"A1": self.a, "A2": self.a}))["runReset"], True)

    def testA1CornerIsInStateAndPayload(self):
        self.assertEqual(self.s.getPlanPayload()["a1Corner"], "auto")
        self.s.setA1Corner("bottomLeft")
        self.assertEqual(self.s.getState()["plate"]["a1Corner"], "bottomLeft")
        self.assertEqual(self.s.getPlanPayload()["a1Corner"], "bottomLeft")
        with self.assertRaises(ValueError):
            self.s.setA1Corner("left")

    def testPayload(self):
        self.s.savePlan([add(self.a, "A2", "A1"), wait(30), add(self.b, "B1", "A1")])
        p = self.s.getPlanPayload()
        self.assertEqual(p["wells"], {"A1": [self.a, self.b], "A2": [self.a], "B1": [self.b]})
        self.assertEqual(p["steps"][1], wait(30))
        self.assertEqual(p["plate"], {"wells": 96, "rows": 8, "columns": 12})
        self.assertEqual([c["id"] for c in p["compounds"]], self.ids)

    def testStartNeedsPlanAndClearsOldRun(self):
        with self.assertRaises(ValueError):
            self.s.startRun()
        self.startRun({"A1": self.a}, active=self.a)
        self.s.reportDrop(["B2"])
        self.s.startRun()
        st = self.s.getState()
        self.assertEqual((st["events"], st["run"]["alarm"], st["run"]["status"]), ([], None, "active"))

    def testDropNeedsStartedRunAndCompound(self):
        self.s.savePlan(S({"A1": self.a}))
        self.assertEqual(self.s.reportDrop(["A1"])["reason"], "runNotStarted")
        self.s.startRun()
        self.assertEqual(self.s.reportDrop(["A1"])["reason"], "noActiveCompound")

    def testCorrectDropsAndCompletion(self):
        self.startRun({"A1": self.a, "A2": self.a}, active=self.a)
        self.assertFalse(self.s.reportDrop(["A1"])["complete"])
        res = self.s.reportDrop(["A2"])
        self.assertTrue(res["complete"])
        self.assertEqual(self.s.getState()["run"]["status"], "complete")

    def testWrongWellRaisesAlarmUntilAcknowledged(self):
        self.startRun({"A1": self.a, "B1": self.b}, active=self.a)
        res = self.s.reportDrop(["B1"])
        self.assertEqual((res["results"][0]["kind"], res["alarm"]["wells"]), ("wrong", ["B1"]))
        self.s.reportDrop(["C1"])  # a second problem joins the same alarm
        self.assertEqual(self.s.getState()["run"]["alarm"]["wells"], ["B1", "C1"])
        self.s.acknowledgeAlarm()
        self.assertIsNone(self.s.getState()["run"]["alarm"])

    def testAcknowledgeCompletesARunThatFinishedWithAnAlarm(self):
        self.startRun({"A1": self.a}, active=self.a)
        self.s.reportDrop(["B1"])
        self.s.reportDrop(["A1"])
        self.assertEqual(self.s.getState()["run"]["status"], "active")
        self.s.acknowledgeAlarm()
        self.assertEqual(self.s.getState()["run"]["status"], "complete")

    def testDuplicateAfterCompletionIsStillChecked(self):
        self.startRun({"A1": self.a}, active=self.a)
        self.s.reportDrop(["A1"])
        self.assertEqual(self.s.reportDrop(["A1"])["results"][0]["kind"], "duplicate")

    def testExplicitCompoundBeatsActiveOne(self):
        self.startRun({"A1": self.a}, active=self.b)
        self.assertEqual(self.s.reportDrop(["A1"], self.a)["results"][0]["kind"], "ok")

    def testWellOffThePlateIsAnErrorAndWritesNothing(self):
        self.startRun({"A1": self.a}, active=self.a)
        with self.assertRaises(ValueError):
            self.s.reportDrop(["A1", "A13"])
        st = self.s.getState()
        self.assertEqual((st["events"], st["run"]["done"]), ([], {}))

    def testStatePersistsAcrossRestart(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            s1 = PlateStore(path)
            cid = s1.listCompounds()[0]["id"]
            s1.savePlan(S({"A1": cid}))
            s1.setA1Corner("topRight")
            s1.startRun()
            s1.setActiveCompound(cid)
            s1.reportDrop(["B2"])
            s1.conn.close()
            s2 = PlateStore(path)
            st = s2.getState()
            self.assertEqual((st["plate"]["steps"], st["plate"]["a1Corner"]), ([add(cid, "A1")], "topRight"))
            self.assertEqual(st["run"]["alarm"]["wells"], ["B2"])
            s2.conn.close()

    def testEachWellsTimerStartsWhenItsCompoundGoesIn(self):
        self.s.savePlan([add(self.a, "A1", "B1"), wait(60), add(self.b, "A1", "B1")])
        self.s.startRun()
        self.assertEqual(self.s.getState()["run"]["wellTimers"]["A1"][0]["state"], "pending")
        self.s.reportDrop(["A1"], self.a)
        timers = self.s.getState()["run"]["wellTimers"]
        self.assertEqual((timers["A1"][0]["state"], timers["B1"][0]["state"]), ("running", "pending"))
        self.assertAlmostEqual(timers["A1"][0]["endsAt"] - time.time(), 60, delta=2)
        res = self.s.reportDrop(["A1"], self.b)  # too early
        self.assertEqual((res["results"][0]["kind"], res["alarm"]["kind"]), ("early", "early"))

    def testNextWellByHand(self):
        with self.assertRaises(ValueError):
            self.s.markWellDone("A1")  # run not started
        self.s.savePlan([add(self.a, "A1", "B1"), wait(3600), add(self.b, "A1")])
        self.s.startRun()
        self.s.setActiveCompound(self.a)
        self.s.reportDrop(["A1"])
        res = self.s.markWellDone("b1")  # the camera missed B1
        self.assertEqual((res["results"][0]["well"], res["results"][0]["kind"]), ("B1", "ok"))
        with self.assertRaises(ValueError):
            self.s.markWellDone("B1")  # has everything it needs
        st = self.s.getState()
        self.assertEqual((st["run"]["progress"]["done"], st["run"]["wellTimers"]["A1"][0]["state"]), (2, "running"))
        self.assertEqual((st["events"][0]["well"], st["events"][0]["source"]), ("B1", "manual"))
        with self.assertRaises(ValueError) as cm:
            self.s.markWellDone("A1")  # A1 needs B next, but its timer runs
        self.assertIn("timer", str(cm.exception))

    def testOneCompoundPerWellDatabaseIsConverted(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            s1 = PlateStore(path)
            a, b = s1.listCompounds()[0]["id"], s1.listCompounds()[1]["id"]
            s1.conn.execute("UPDATE plate SET plan = ? WHERE id = 1", (json.dumps({"B1": b, "A1": a}),))
            s1.conn.commit()
            s1.conn.close()
            s2 = PlateStore(path)
            self.assertEqual(s2.getPlate()["plan"], [add(a, "A1"), add(b, "B1")])
            s2.conn.close()

    def testOldDatabaseIsRefusedWithAClearMessage(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.db")
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE plate (id INTEGER PRIMARY KEY, size INTEGER, status TEXT)")
            conn.close()
            with self.assertRaises(RuntimeError) as cm:
                PlateStore(path)
            self.assertIn("older version", str(cm.exception))


# ======================================================================== 3. server
def freePort():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ServerTests(unittest.TestCase):
    TOKEN = "testToken"

    @classmethod
    def setUpClass(cls):
        import app as appModule
        import uvicorn

        cls.appModule = appModule
        cls.tmp = tempfile.TemporaryDirectory()
        cls.port = freePort()
        cls.app = appModule.createApp(os.path.join(cls.tmp.name, "t.db"), apiToken=cls.TOKEN)
        config = uvicorn.Config(cls.app, host="127.0.0.1", port=cls.port, log_level="warning",
                                timeout_graceful_shutdown=1)  # fmt: skip
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(5)
        cls.app.state.store.conn.close()
        cls.tmp.cleanup()

    def setUp(self):
        # every test starts from: demo compounds, empty plan, no run
        self.call("PUT", "/api/plan", {"plan": []})
        self.call("POST", "/api/run/reset")
        self.call("POST", "/api/activeCompound", {"compoundId": None})
        self.call("POST", "/api/a1Corner", {"a1Corner": "auto"})
        self.ids = [c["id"] for c in self.call("GET", "/api/state")[1]["compounds"]]

    def call(self, method, path, body=None, token=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Api-Token"] = token
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def startRun(self, plan, active=None):
        self.call("PUT", "/api/plan", {"plan": S(plan)})
        self.call("POST", "/api/run/start", {})
        if active is not None:
            self.call("POST", "/api/activeCompound", {"compoundId": active})

    def testPagesAreServed(self):
        for path, needle in [("/", b"cornerSelect"), ("/static/app.js", b"EventSource"),
                             ("/static/style.css", b"color-scheme: light"), ("/camera", b"Camera Simulator"),
                             ("/static/camera.js", b"/api/sim/drop")]:  # fmt: skip
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
                self.assertEqual((r.status, r.headers.get("Cache-Control")), (200, "no-cache"))
                self.assertIn(needle, r.read())

    def testSimulatorDropsAreJudgedLikeCameraDropsWithoutToken(self):
        cid = self.call("GET", "/api/state")[1]["compounds"][0]["id"]
        self.startRun({"A1": cid}, cid)
        status, res = self.call("POST", "/api/sim/drop", {"wells": ["A2"]})  # no token: the page has none
        self.assertEqual((status, res["results"][0]["kind"]), (200, "unplanned"))
        self.assertEqual(self.call("POST", "/api/sim/drop", {"wells": ["A1"]})[1]["results"][0]["kind"], "ok")
        st = self.call("GET", "/api/state")[1]
        self.assertEqual([e["source"] for e in st["events"]], ["simulator", "simulator"])
        self.assertEqual(self.call("POST", "/api/sim/heartbeat", {})[0], 200)
        self.assertTrue(self.call("GET", "/api/state")[1]["camera"]["online"])

    def testTipIsJudgedLiveForTheCompoundInTheTip(self):
        comps = self.call("GET", "/api/state")[1]["compounds"]
        a, b = comps[0]["id"], comps[1]["id"]
        self.startRun({"A1": a, "B1": b}, a)
        tipOf = lambda: self.call("GET", "/api/state")[1]["tip"]  # noqa: E731
        self.call("POST", "/api/sim/tip", {"well": "A1"})
        self.assertEqual((tipOf()["well"], tipOf()["kind"]), ("A1", "ok"))
        self.call("POST", "/api/sim/drop", {"wells": ["A1"]})
        self.assertEqual(tipOf()["kind"], "added")  # still over the well it just filled: not a duplicate
        for well, kind in [("B1", "wrong"), ("H12", "unplanned"), ("A1", "duplicate")]:  # left and came back
            self.call("POST", "/api/sim/tip", {"well": well})
            self.assertEqual(tipOf()["kind"], kind, well)
        self.call("POST", "/api/sim/tip", {"well": None})
        self.assertEqual(tipOf(), {"well": None})
        status, _ = self.call("POST", "/api/camera/tip", {"well": "A1"})  # the camera route needs the token
        self.assertEqual(status, 401)
        self.assertEqual(self.call("POST", "/api/camera/tip", {"well": "A1"}, token=self.TOKEN)[0], 200)
        self.assertEqual(self.call("POST", "/api/sim/tip", {"well": "Z9"})[0], 400)

    def testHealthAndStateShape(self):
        status, health = self.call("GET", "/api/health")
        self.assertEqual((status, health["ok"], health["tokenRequired"]), (200, True, True))
        st = self.call("GET", "/api/state")[1]
        self.assertEqual(set(st), {"compounds", "plate", "run", "events", "camera", "tip", "serverId", "revision"})
        self.assertEqual((st["plate"]["rows"], st["plate"]["columns"]), (8, 12))

    def testCompoundCrudAndErrors(self):
        status, res = self.call("POST", "/api/compounds", {"name": "Water", "volumeUl": 15, "color": "#112233"})
        self.assertEqual(status, 201)
        cid = res["compound"]["id"]
        self.assertIn(cid, [c["id"] for c in res["state"]["compounds"]])  # every change returns the new state
        self.assertEqual(self.call("PUT", f"/api/compounds/{cid}", {"name": "W", "volumeUl": 1, "color": "#112233"})[0],
                         200)  # fmt: skip
        self.assertEqual(self.call("DELETE", f"/api/compounds/{cid}")[0], 200)
        self.assertEqual(self.call("DELETE", f"/api/compounds/{cid}")[0], 404)
        self.assertEqual(self.call("POST", "/api/compounds", {"name": ""})[0], 400)

    def testBadBodiesAreCleanErrors(self):
        status, res = self.call("PUT", "/api/plan", raw=b"{not json")
        self.assertEqual((status, "error" in res), (400, True))
        self.assertEqual(self.call("POST", "/api/a1Corner", {"a1Corner": "middle"})[0], 400)
        self.assertEqual(self.call("GET", "/api/nothing")[0], 404)

    def testPlanAndCornerReachThePayload(self):
        a = self.ids[0]
        self.call("PUT", "/api/plan", {"plan": [add(a, "a1")]})
        self.call("POST", "/api/a1Corner", {"a1Corner": "bottomRight"})
        p = self.call("GET", "/api/plan")[1]
        self.assertEqual((p["wells"], p["a1Corner"]), ({"A1": [a]}, "bottomRight"))

    def testDropNeedsTokenButTestDropDoesNot(self):
        a = self.ids[0]
        self.startRun({"A1": a, "A2": a}, active=a)
        self.assertEqual(self.call("POST", "/api/drop", {"wells": ["A1"]})[0], 401)
        self.assertEqual(self.call("POST", "/api/drop", {"wells": ["A1"]}, token="wrong")[0], 401)
        status, res = self.call("POST", "/api/drop", {"wells": ["A1"]}, token=self.TOKEN)
        self.assertEqual((status, res["results"][0]["kind"]), (200, "ok"))
        status, res = self.call("POST", "/api/testDrop", {"wells": ["A2"]})
        self.assertEqual((status, res["complete"]), (200, True))
        self.assertEqual([e["source"] for e in res["state"]["events"]], ["test", "camera"])

    def testWrongWellAndRefusalsAndBadWell(self):
        a, b = self.ids[0], self.ids[1]
        self.call("PUT", "/api/plan", {"plan": S({"A1": a, "B1": b})})
        status, res = self.call("POST", "/api/drop", {"well": "B1"}, token=self.TOKEN)
        self.assertEqual((status, res["accepted"], res["reason"]), (200, False, "runNotStarted"))
        self.startRun({"A1": a, "B1": b}, active=a)
        res = self.call("POST", "/api/drop", {"well": "B1"}, token=self.TOKEN)[1]
        self.assertEqual(res["alarm"]["kind"], "wrong")
        self.assertEqual(self.call("POST", "/api/drop", {"wells": ["A13"]}, token=self.TOKEN)[0], 400)
        self.call("POST", "/api/alarm/ack")
        self.assertIsNone(self.call("GET", "/api/state")[1]["run"]["alarm"])

    def testHeartbeatMakesCameraOnlineAndNeedsToken(self):
        self.assertEqual(self.call("POST", "/api/camera/heartbeat", {})[0], 401)
        self.assertEqual(self.call("POST", "/api/camera/heartbeat", {}, token=self.TOKEN)[0], 200)
        cam = self.call("GET", "/api/state")[1]["camera"]
        self.assertTrue(cam["online"])
        link = self.appModule.CameraLink()
        link.touch()
        link.lastSeenMono -= self.appModule.CAMERA_ONLINE_SECONDS + 1
        self.assertFalse(link.online())

    # ---- live stream
    def readEvent(self, resp, timeout=5):
        """Read one 'event: x / data: y' block from the SSE stream."""
        end = time.time() + timeout
        name, data = None, None
        while time.time() < end:
            line = resp.readline().decode().rstrip("\r\n")
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
            elif line == "" and name:
                return name, data
        self.fail(f"no event within {timeout} s")

    def testLiveStreamPushesAlarmToOpenPages(self):
        a, b = self.ids[0], self.ids[1]
        self.startRun({"A1": a, "B1": b}, active=a)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", "/api/stream")
            resp = conn.getresponse()
            self.assertIn("text/event-stream", resp.getheader("Content-Type"))
            name, first = self.readEvent(resp)
            self.assertEqual(name, "state")
            self.assertIsNone(first["run"]["alarm"])  # a new page gets the full state at once
            self.call("POST", "/api/drop", {"wells": ["B1"]}, token=self.TOKEN)  # camera sees a mistake
            alarm = None
            for _ in range(5):  # skip the 'camera came online' update if it arrives first
                _, st = self.readEvent(resp)
                if st["run"]["alarm"]:
                    alarm = st["run"]["alarm"]
                    break
            self.assertEqual((alarm["kind"], alarm["wells"]), ("wrong", ["B1"]))
        finally:
            conn.close()

    def testSlowClientIsCutLoose(self):
        hub = self.appModule.Hub()
        q = hub.subscribe()
        for _ in range(150):  # nobody reads the queue
            hub.publish("state", {"n": 1})
        self.assertNotIn(q, hub.clients)
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        self.assertEqual(items, [None])  # the stream ends; the browser reconnects and resyncs

    # ---- the fake camera, against the real server
    def testFakeCameraFinishesTheRunAndItsMistakesRaiseAlarms(self):
        import simulateCamera as cam

        a, b = self.ids[0], self.ids[1]
        self.call("PUT", "/api/plan", {"plan": [add(a, "A1", "A2"), wait(1), add(b, "A1", "B1")]})  # A1 gets both
        self.call("POST", "/api/run/start", {})
        client = cam.WellPlateClient(f"http://127.0.0.1:{self.port}", self.TOKEN)
        stopAck, acked = threading.Event(), []

        def acker():  # plays the part of the person pressing Acknowledge
            while not stopAck.is_set():
                alarm = self.call("GET", "/api/state")[1]["run"]["alarm"]
                if alarm:
                    acked.append(alarm["kind"])
                    self.call("POST", "/api/alarm/ack")
                time.sleep(0.02)

        t = threading.Thread(target=acker, daemon=True)
        t.start()
        try:
            finished = cam.simulate(client, delay=0, errorRate=1.0, pollSeconds=0.02, rng=random.Random(1))
        finally:
            stopAck.set()
            t.join(2)
        self.assertTrue(finished)
        st = self.call("GET", "/api/state")[1]
        self.assertEqual((st["run"]["status"], st["run"]["progress"]), ("complete", {"done": 4, "total": 4}))
        self.assertEqual(len(acked), 4)  # one mistake per addition (wrong, order or duplicate), all caught

    def testFakeCameraErrorsAreReadable(self):
        import simulateCamera as cam

        client = cam.WellPlateClient(f"http://127.0.0.1:{self.port}", "wrongToken")
        with self.assertRaises(cam.ApiError) as cm:
            client.heartbeat()
        self.assertEqual(cm.exception.status, 401)


if __name__ == "__main__":
    unittest.main(verbosity=1)
