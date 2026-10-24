"""Well Plate UI: pure logic.

No web server and no database in this file, so it is easy to test and easy to
read. It answers three questions:

  * Is this well id valid for the plate?             (parseWell, inPlate)
  * Is this compound / plan acceptable?              (validateCompound, cleanSteps)
  * A tip just entered these wells. Was that right?  (classifyDrops)

One plate type only: 96 wells (8 rows x 12 columns), one tip at a time. That is
what the camera tracker supports.

Naming: camelCase for functions and variables, PascalCase for classes,
UPPER_CASE for constants.
"""

import re

ROWS, COLUMNS = 8, 12
WELL_RE = re.compile(r"^([A-Z])([0-9]{1,2})$")
COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
# Where well A1 is in the camera picture. The camera code turns it into the
# direction rows A -> H run in the image. "auto" lets the camera guess.
A1_CORNERS = ("auto", "topLeft", "topRight", "bottomLeft", "bottomRight")

# A sandwich ELISA (e.g. IL-6), with typical volumes per well.
DEMO_COMPOUNDS = [
    {"name": "IL-6 standard", "tag": "STD", "volumeUl": 100, "color": "#2F6FDE"},
    {"name": "Serum sample", "tag": "SMP", "volumeUl": 100, "color": "#E5484D"},
    {"name": "Assay diluent (blank)", "tag": "BLK", "volumeUl": 100, "color": "#7A8B99"},
    {"name": "Detection antibody", "tag": "DET", "volumeUl": 100, "color": "#8E5CD9"},
    {"name": "Streptavidin-HRP", "tag": "HRP", "volumeUl": 100, "color": "#E26FB0"},
    {"name": "TMB substrate", "tag": "TMB", "volumeUl": 100, "color": "#17A2B8"},
    {"name": "Stop solution (2N H2SO4)", "tag": "STOP", "volumeUl": 50, "color": "#F5A524"},
]


class NotFoundError(LookupError):
    """Something the caller referred to does not exist (becomes HTTP 404)."""


# ---------------------------------------------------------------- wells
def parseWell(wellId):
    """'b4' -> (1, 3) as (row, column), both zero-based."""
    m = WELL_RE.match(str(wellId).strip().upper())
    if not m:
        raise ValueError(f"Bad well id: {wellId!r}. Use a letter and a number, like B4.")
    return ord(m.group(1)) - 65, int(m.group(2)) - 1


def wellName(row, col):
    return chr(65 + row) + str(col + 1)


def normalizeWell(wellId):
    """'b04' -> 'B4'"""
    row, col = parseWell(wellId)
    return wellName(row, col)


def inPlate(wellId):
    try:
        row, col = parseWell(wellId)
    except ValueError:
        return False
    return 0 <= row < ROWS and 0 <= col < COLUMNS


def sortWells(wells):
    """Row by row: A1, A2, ... B1, B2, ..."""
    return sorted(wells, key=parseWell)


def normalizeWells(wells):
    """What the camera reports -> clean list of well ids on the plate."""
    if isinstance(wells, str):
        wells = [wells]
    if not isinstance(wells, (list, tuple)) or not wells:
        raise ValueError('Report at least one well, like {"wells": ["B4"]}.')
    if len(wells) > ROWS * COLUMNS:
        raise ValueError("Too many wells in one report.")
    out = []
    for w in wells:
        n = normalizeWell(w)
        if not inPlate(n):
            raise ValueError(f"Well {n} is not on a 96-well plate.")
        if n not in out:
            out.append(n)
    return out


def normalizeA1Corner(corner):
    corner = str(corner or "")
    if corner not in A1_CORNERS:
        raise ValueError(f"a1Corner must be one of: {', '.join(A1_CORNERS)}.")
    return corner


def optionalId(value, label="id"):
    """None stays None; 3 and '3' become 3; anything else is an error."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number.")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None


# ---------------------------------------------------------------- compounds and plan
def validateCompound(data):
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("Enter a compound name.")
    if len(name) > 40:
        raise ValueError("Compound name is too long (40 characters max).")
    try:
        volume = float(data.get("volumeUl"))
    except (TypeError, ValueError):
        raise ValueError("Volume must be a number.") from None
    if not (0 <= volume <= 100000):
        raise ValueError("Volume must be between 0 and 100000 µL.")
    color = str(data.get("color", "")).strip()
    if not COLOR_RE.match(color):
        raise ValueError("Color must look like #2F6FDE.")
    tag = str(data.get("tag", "") or "").strip().upper()
    if not tag:
        tag = re.sub(r"[^A-Za-z0-9]", "", name)[:3].upper()
    return {"name": name, "tag": tag[:4], "volumeUl": volume, "color": color.upper()}


MAX_WAIT_SECONDS = 24 * 3600


def cleanSteps(steps, validIds):
    """The plan is an ordered list of steps:
      {"kind": "add", "compoundId": 3, "wells": ["A1", "B1"]}   add this compound to these wells
      {"kind": "wait", "seconds": 300}                          a timer, run per well (see wellTimers)
    Wells off the plate, unknown compounds and empty steps are dropped."""
    if not isinstance(steps, list):
        raise ValueError('The plan must be a list of steps like [{"kind": "add", "compoundId": 1, "wells": ["A1"]}].')
    clean = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        if st.get("kind") == "wait":
            try:
                seconds = int(st.get("seconds"))
            except (TypeError, ValueError):
                raise ValueError("A timer needs a number of seconds.") from None
            if not 1 <= seconds <= MAX_WAIT_SECONDS:
                raise ValueError("A timer must be between 1 second and 24 hours.")
            clean.append({"kind": "wait", "seconds": seconds})
            continue
        cid = st.get("compoundId")
        if st.get("kind") != "add" or isinstance(cid, bool) or cid not in validIds:
            continue
        wells = []
        for w in st.get("wells") or []:
            try:
                n = normalizeWell(w)
            except ValueError:
                continue
            if inPlate(n) and n not in wells:
                wells.append(n)
        if wells:
            clean.append({"kind": "add", "compoundId": cid, "wells": sortWells(wells)})
    return clean


def stepsFromWellMap(plan, compoundOrder):
    """An older plan {"A1": 3, ...} (one compound per well) as steps, one per compound."""
    return [{"kind": "add", "compoundId": cid, "wells": sortWells(w for w, c in plan.items() if c == cid)}
            for cid in compoundOrder if cid in plan.values()]  # fmt: skip


def wellSequences(steps):
    """{well: [step index, ...]}: what goes into each well, in order."""
    seq = {}
    for i, st in enumerate(steps):
        if st["kind"] == "add":
            for w in st["wells"]:
                seq.setdefault(w, []).append(i)
    return seq


def stepComplete(steps, done, i):
    st = steps[i]
    return st["kind"] == "add" and all(done.get(str(i), {}).get(w) for w in st["wells"])


def doneTime(done, i, well):
    """When step i's compound went into this well (epoch s), or None if it has not. Runs from before per-well
    timers stored True: their time is unknown, so their timers count as long finished (0)."""
    v = done.get(str(i), {}).get(well)
    if not v:
        return None
    return 0.0 if v is True else float(v)


def wellTimers(steps, done, now):
    """{well: [{"step": j, "seconds": s, "state": pending|running|done, "endsAt": epoch s or None}]}.

    Timers run PER WELL: a wait step holds a well that gets a compound before it and after it, and starts when the
    compound before it went into that well (e.g. A, wait 1 min, B: each well's minute starts when its A goes in,
    and only that well has to wait for its B). Two waits in a row run one after the other."""
    out = {}
    for w, mine in wellSequences(steps).items():
        readyAt, last = None, mine[-1]
        for i, st in enumerate(steps[: last + 1]):
            if st["kind"] == "add":
                if w in st["wells"]:
                    readyAt = doneTime(done, i, w)
                continue
            if i < mine[0]:  # nothing in this well yet: this timer does not hold it
                continue
            endsAt = None if readyAt is None else readyAt + st["seconds"]
            state = "pending" if endsAt is None else "done" if now >= endsAt else "running"
            out.setdefault(w, []).append({"step": i, "seconds": st["seconds"], "state": state, "endsAt": endsAt})
            readyAt = endsAt
    return out


def nextStepFor(steps, done, well):
    """Index of the add step this well needs next (None: nothing left, or nothing planned)."""
    return next((i for i in wellSequences(steps).get(well, []) if doneTime(done, i, well) is None), None)


def classifyDrops(wells, compoundId, steps, done, timers, now=0.0):
    """Judge each reported well against the steps.

    kind is one of:
      ok         the next compound planned for this well
      wrong      this well needs a different compound
      unplanned  nothing planned for this well
      duplicate  this compound is already in this well (every time it was planned)
      order      this compound is planned here, but an earlier compound is not in the well yet
      early      this well's timer before this step is still running

    timers is wellTimers(...). An ok drop is stored with its time (`now`): that starts the well's next timer.
    Returns (results, newDone); results carry plannedId (the compound the well needs next) and step.
    """
    seq = wellSequences(steps)
    newDone = {k: dict(v) for k, v in done.items()}
    results = []
    for w in wells:
        mine = seq.get(w, [])
        pending = [i for i in mine if not newDone.get(str(i), {}).get(w)]
        plannedId = steps[pending[0]]["compoundId"] if pending else None
        match = next((i for i in pending if steps[i]["compoundId"] == compoundId), None)
        step = None
        if not mine:
            kind = "unplanned"
        elif match is None:
            kind = "duplicate" if any(steps[i]["compoundId"] == compoundId for i in mine) else "wrong"
        elif match != pending[0]:
            kind = "order"
        elif any(t["step"] < match and t["state"] != "done" for t in timers.get(w, [])):
            kind = "early"
        else:
            kind, step = "ok", match
            newDone.setdefault(str(match), {})[w] = now or True
        results.append({"well": w, "kind": kind, "plannedId": plannedId, "step": step})
    return results, newDone


def alarmMessage(kind, well, plannedName, detectedName):
    if kind == "wrong":
        return f"{well} needs {plannedName} next, but {detectedName} was detected."
    if kind == "unplanned":
        return f"{well} has no compound planned, but {detectedName} was detected."
    if kind == "order":
        return f"{well} needs {plannedName} before {detectedName}."
    if kind == "early":
        return f"{detectedName} was added to {well} before the timer finished."
    return f"{well} already has {detectedName}. A second drop was detected."


def mergeAlarm(existing, badResults, message, nowIso):
    """First problem sets the headline; later ones are added to the same alarm."""
    wells = [r["well"] for r in badResults]
    if existing:
        merged = dict(existing)
        merged["wells"] = sortWells(set(existing["wells"]) | set(wells))
        merged["count"] = existing.get("count", 1) + len(wells)
        return merged
    return {
        "kind": badResults[0]["kind"],
        "well": badResults[0]["well"],  # the well the message is about
        "wells": sortWells(set(wells)),
        "message": message,
        "count": len(wells),
        "time": nowIso,
    }


def progress(steps, done):
    """(wells done, wells planned), counting each compound in each well once."""
    total = sum(len(st["wells"]) for st in steps if st["kind"] == "add")
    finished = sum(1 for i, st in enumerate(steps) if st["kind"] == "add" for w in st["wells"]
                   if done.get(str(i), {}).get(w))  # fmt: skip
    return finished, total


def isComplete(steps, done):
    adds = [i for i, st in enumerate(steps) if st["kind"] == "add"]
    return bool(adds) and all(stepComplete(steps, done, i) for i in adds)
