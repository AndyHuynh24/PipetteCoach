"""Well Plate UI: storage.

One SQLite file holds everything: the compounds, the plan (an ordered list of
steps: add a compound to wells, or wait), the state of the current run, and the
log of detected drops. All rules live in plateLogic.py;
this file only reads and writes.

Every public method is one transaction, so the browser and the camera can call
at the same time without corrupting each other.
"""

import contextlib
import datetime
import json
import os
import sqlite3
import threading
import time

import plateLogic as pl

SCHEMA = """
CREATE TABLE IF NOT EXISTS compounds (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL,
    tag       TEXT    NOT NULL,
    volumeUl  REAL    NOT NULL,
    color     TEXT    NOT NULL,
    position  INTEGER NOT NULL,
    created   TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS plate (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    status            TEXT    NOT NULL,
    plan              TEXT    NOT NULL,
    done              TEXT    NOT NULL,
    alarm             TEXT,
    activeCompoundId  INTEGER,
    a1Corner          TEXT    NOT NULL DEFAULT 'auto',
    startedAt         REAL,
    stepDoneAt        TEXT    NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    wellId        TEXT    NOT NULL,
    plannedId     INTEGER,
    plannedName   TEXT    NOT NULL,
    detectedId    INTEGER,
    detectedName  TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


def nowIso():
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")


class PlateStore:
    def __init__(self, path=":memory:", seedDemo=True):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        with self._tx() as c:
            c.executescript(SCHEMA)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(plate)")}
            if "size" in cols:
                raise RuntimeError(f"{path} was made by an older version of this app. Move it away to start fresh.")
            if "startedAt" not in cols:  # made before timers existed
                c.execute("ALTER TABLE plate ADD COLUMN startedAt REAL")
                c.execute("ALTER TABLE plate ADD COLUMN stepDoneAt TEXT NOT NULL DEFAULT '{}'")
        with self._tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO plate(id, status, plan, done, alarm, activeCompoundId)"
                " VALUES (1, 'draft', '[]', '{}', NULL, NULL)"
            )
            plan = json.loads(c.execute("SELECT plan FROM plate WHERE id = 1").fetchone()["plan"])
            if isinstance(plan, dict):  # made before a well could hold several compounds
                steps = pl.stepsFromWellMap(plan, [x["id"] for x in self._readCompounds(c)])
                c.execute("UPDATE plate SET plan = ? WHERE id = 1", (json.dumps(steps),))
                self._resetRunColumns(c)
            if not c.execute("SELECT 1 FROM meta WHERE key = 'seeded'").fetchone():
                # Demo compounds appear once, on the very first start. Deleting
                # them all later does not bring them back.
                if seedDemo:
                    for i, comp in enumerate(pl.DEMO_COMPOUNDS):
                        c.execute(
                            "INSERT INTO compounds(name, tag, volumeUl, color, position, created)"
                            " VALUES (?, ?, ?, ?, ?, ?)",
                            (comp["name"], comp["tag"], comp["volumeUl"], comp["color"], i, nowIso()),
                        )
                c.execute("INSERT INTO meta(key, value) VALUES ('seeded', '1')")

    @contextlib.contextmanager
    def _tx(self):
        with self.lock:
            with self.conn:  # commits on success, rolls back on any exception
                yield self.conn

    # ------------------------------------------------------------ row helpers
    @staticmethod
    def _compound(row):
        return {k: row[k] for k in ("id", "name", "tag", "volumeUl", "color", "position", "created")}

    @staticmethod
    def _plate(row):
        return {
            "status": row["status"],
            "plan": json.loads(row["plan"]),
            "done": json.loads(row["done"]),
            "alarm": json.loads(row["alarm"]) if row["alarm"] else None,
            "activeCompoundId": row["activeCompoundId"],
            "a1Corner": row["a1Corner"],
            "startedAt": row["startedAt"],
            "stepDoneAt": json.loads(row["stepDoneAt"]),
        }

    def _readCompounds(self, c):
        rows = c.execute("SELECT * FROM compounds ORDER BY position, id").fetchall()
        return [self._compound(r) for r in rows]

    def _readPlate(self, c):
        return self._plate(c.execute("SELECT * FROM plate WHERE id = 1").fetchone())

    @staticmethod
    def _resetRunColumns(c):
        c.execute(
            "UPDATE plate SET done = '{}', stepDoneAt = '{}', startedAt = NULL, alarm = NULL, status = 'draft'"
            " WHERE id = 1"
        )

    # ------------------------------------------------------------ compounds
    def listCompounds(self):
        with self._tx() as c:
            return self._readCompounds(c)

    def saveCompound(self, data, compoundId=None):
        """Create (compoundId=None) or update one compound."""
        clean = pl.validateCompound(data)
        with self._tx() as c:
            if compoundId is None:
                position = c.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM compounds").fetchone()[0]
                cur = c.execute(
                    "INSERT INTO compounds(name, tag, volumeUl, color, position, created) VALUES (?, ?, ?, ?, ?, ?)",
                    (clean["name"], clean["tag"], clean["volumeUl"], clean["color"], position, nowIso()),
                )
                compoundId = cur.lastrowid
            else:
                cur = c.execute(
                    "UPDATE compounds SET name = ?, tag = ?, volumeUl = ?, color = ? WHERE id = ?",
                    (clean["name"], clean["tag"], clean["volumeUl"], clean["color"], compoundId),
                )
                if cur.rowcount == 0:
                    raise pl.NotFoundError("That compound no longer exists.")
            return self._compound(c.execute("SELECT * FROM compounds WHERE id = ?", (compoundId,)).fetchone())

    def deleteCompound(self, compoundId):
        """Remove a compound and its steps."""
        with self._tx() as c:
            if not c.execute("SELECT 1 FROM compounds WHERE id = ?", (compoundId,)).fetchone():
                raise pl.NotFoundError("That compound no longer exists.")
            plate = self._readPlate(c)
            mine = [st for st in plate["plan"] if st["kind"] == "add" and st["compoundId"] == compoundId]
            wells = sum(len(st["wells"]) for st in mine)
            c.execute("DELETE FROM compounds WHERE id = ?", (compoundId,))
            runReset = False
            if mine:
                keep = [st for st in plate["plan"] if st not in mine]
                c.execute("UPDATE plate SET plan = ? WHERE id = 1", (json.dumps(keep),))
                if plate["status"] in ("active", "complete"):
                    self._resetRunColumns(c)
                    runReset = True
            if plate["activeCompoundId"] == compoundId:
                c.execute("UPDATE plate SET activeCompoundId = NULL WHERE id = 1")
            return {"wells": wells, "runReset": runReset}

    # ------------------------------------------------------------ plan
    def getPlate(self):
        with self._tx() as c:
            return self._readPlate(c)

    def savePlan(self, steps):
        """Replace the whole plan (the list of steps). Wells off the plate and
        unknown compounds are dropped. Changing the plan during a run resets the run."""
        with self._tx() as c:
            validIds = {r["id"] for r in c.execute("SELECT id FROM compounds")}
            clean = pl.cleanSteps(steps, validIds)
            old = self._readPlate(c)
            runReset = clean != old["plan"] and old["status"] in ("active", "complete")
            c.execute("UPDATE plate SET plan = ? WHERE id = 1", (json.dumps(clean),))
            if runReset:
                self._resetRunColumns(c)
            return {"runReset": runReset, "steps": len(clean)}

    def setActiveCompound(self, compoundId):
        """The compound the user says is in the tip right now. The camera sees
        WHERE the tip goes, not WHAT it holds, so that comes from here."""
        with self._tx() as c:
            if compoundId is not None:
                if not c.execute("SELECT 1 FROM compounds WHERE id = ?", (compoundId,)).fetchone():
                    raise pl.NotFoundError("Unknown compound.")
            c.execute("UPDATE plate SET activeCompoundId = ? WHERE id = 1", (compoundId,))

    def setA1Corner(self, corner):
        """Where well A1 is in the camera picture (the camera code reads it from the plan)."""
        corner = pl.normalizeA1Corner(corner)
        with self._tx() as c:
            c.execute("UPDATE plate SET a1Corner = ? WHERE id = 1", (corner,))

    # ------------------------------------------------------------ run control
    def startRun(self):
        with self._tx() as c:
            if not any(st["kind"] == "add" for st in self._readPlate(c)["plan"]):
                raise ValueError("Nothing is planned. Paint some wells first.")
            c.execute("DELETE FROM events")
            self._resetRunColumns(c)
            c.execute("UPDATE plate SET status = 'active', startedAt = ? WHERE id = 1", (time.time(),))

    def resetRun(self):
        with self._tx() as c:
            c.execute("DELETE FROM events")
            self._resetRunColumns(c)

    def acknowledgeAlarm(self):
        """Clear the warning. If that was the only thing standing between the
        run and 'complete', the run completes now."""
        with self._tx() as c:
            plate = self._readPlate(c)
            status = plate["status"]
            if status == "active" and pl.isComplete(plate["plan"], plate["done"]):
                status = "complete"
            c.execute("UPDATE plate SET alarm = NULL, status = ? WHERE id = 1", (status,))

    def markWellDone(self, well):
        """The "Next well" button overrides the camera: the well counts as dispensed with the compound it needs
        next, exactly like a detected drop (logged with source "manual"). Refused while its timer runs."""
        well = pl.normalizeWell(well)
        plate = self.getPlate()
        if plate["status"] != "active":
            raise ValueError("Press Start run first." if plate["status"] == "draft" else "Every well is done.")
        steps, done = plate["plan"], plate["done"]
        step = pl.nextStepFor(steps, done, well)
        if step is None:
            raise ValueError(f"{well} has everything it needs.")
        cid = steps[step]["compoundId"]
        r = pl.classifyDrops([well], cid, steps, done, pl.wellTimers(steps, done, time.time()))[0][0]
        if r["kind"] != "ok":
            raise ValueError(
                f"{well}: its timer is still running." if r["kind"] == "early" else f"{well} is not ready."
            )
        return self.reportDrop([well], cid, source="manual")

    def reportDrop(self, wells, compoundId=None, source="camera"):
        """A tip dispensed into these wells. compoundId=None means "whatever the
        user has selected as active"."""
        source = str(source or "camera")[:20]
        with self._tx() as c:
            plate = self._readPlate(c)
            # 'complete' still checks drops: a tip entering a filled well after
            # the last planned well is done is still a mistake worth flagging.
            if plate["status"] not in ("active", "complete"):
                return {"accepted": False, "reason": "runNotStarted"}
            wells = pl.normalizeWells(wells)
            cid = compoundId if compoundId is not None else plate["activeCompoundId"]
            if cid is None:
                return {"accepted": False, "reason": "noActiveCompound"}
            names = {x["id"]: x["name"] for x in self._readCompounds(c)}
            if cid not in names:
                raise pl.NotFoundError("Unknown compound.")

            steps, nowS = plate["plan"], time.time()
            timers = pl.wellTimers(steps, plate["done"], nowS)
            results, newDone = pl.classifyDrops(wells, cid, steps, plate["done"], timers, nowS)  # stores drop times
            now = nowIso()
            for r in results:
                c.execute(
                    "INSERT INTO events(wellId, plannedId, plannedName, detectedId, detectedName,"
                    " kind, source, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (r["well"], r["plannedId"], names.get(r["plannedId"], "none"), cid, names[cid], r["kind"],
                     source, now),
                )  # fmt: skip
            bad = [r for r in results if r["kind"] != "ok"]
            alarm = plate["alarm"]
            if bad:
                first = bad[0]
                message = pl.alarmMessage(first["kind"], first["well"], names.get(first["plannedId"], "?"), names[cid])
                alarm = pl.mergeAlarm(alarm, bad, message, now)
            complete = pl.isComplete(steps, newDone)
            status = "complete" if complete and alarm is None else plate["status"]
            c.execute(
                "UPDATE plate SET done = ?, alarm = ?, status = ? WHERE id = 1",
                (json.dumps(newDone), json.dumps(alarm) if alarm else None, status),
            )
            return {
                "accepted": True,
                "results": [
                    {"well": r["well"], "kind": r["kind"], "planned": names.get(r["plannedId"], "none")}
                    for r in results
                ],
                "alarm": alarm,
                "complete": complete,
            }

    def judgeTip(self, well):
        """What a drop into `well` would be right now (the tip is over it), without recording anything.
        None when no run is going or no compound is in the tip."""
        with self._tx() as c:
            plate = self._readPlate(c)
            cid = plate["activeCompoundId"]
            if plate["status"] not in ("active", "complete") or cid is None:
                return None
            steps = plate["plan"]
            r = pl.classifyDrops([well], cid, steps, plate["done"], pl.wellTimers(steps, plate["done"], time.time()))[
                0
            ][0]
            names = {x["id"]: x["name"] for x in self._readCompounds(c)}
            return {"kind": r["kind"], "planned": names.get(r["plannedId"]), "compound": names.get(cid)}

    # ------------------------------------------------------------ reading everything
    def getState(self, eventLimit=100):
        """Everything the browser needs to draw itself."""
        with self._tx() as c:
            plate = self._readPlate(c)
            steps, nowS = plate["plan"], time.time()
            done, total = pl.progress(steps, plate["done"])
            evRows = c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (int(eventLimit),)).fetchall()
            return {
                "compounds": self._readCompounds(c),
                "plate": {"rows": pl.ROWS, "columns": pl.COLUMNS, "steps": steps, "a1Corner": plate["a1Corner"]},
                "run": {
                    "status": plate["status"],
                    "activeCompoundId": plate["activeCompoundId"],
                    "done": plate["done"],
                    "alarm": plate["alarm"],
                    "progress": {"done": done, "total": total},
                    "wellTimers": pl.wellTimers(steps, plate["done"], nowS) if plate["status"] != "draft" else {},
                    "serverTime": nowS,
                    "startedAt": plate["startedAt"],
                },
                "events": [
                    {
                        "id": e["id"],
                        "time": e["timestamp"],
                        "well": e["wellId"],
                        "plannedName": e["plannedName"],
                        "detectedName": e["detectedName"],
                        "kind": e["kind"],
                        "source": e["source"],
                    }
                    for e in evRows
                ],
            }

    def getPlanPayload(self):
        """The contract with the camera code: what should go where."""
        with self._tx() as c:
            plate = self._readPlate(c)
            return {
                "plate": {"wells": pl.ROWS * pl.COLUMNS, "rows": pl.ROWS, "columns": pl.COLUMNS},
                "a1Corner": plate["a1Corner"],
                "status": plate["status"],
                "activeCompoundId": plate["activeCompoundId"],
                "compounds": [
                    {"id": x["id"], "name": x["name"], "tag": x["tag"], "volumeUl": x["volumeUl"], "color": x["color"]}
                    for x in self._readCompounds(c)
                ],
                "steps": plate["plan"],
                "wells": {
                    w: [plate["plan"][i]["compoundId"] for i in seq]
                    for w, seq in sorted(pl.wellSequences(plate["plan"]).items(), key=lambda kv: pl.parseWell(kv[0]))
                },  # fmt: skip
            }
