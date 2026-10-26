# Well Plate UI

A web page for planning which substance goes into which well of a 96-well plate, and for
catching mistakes while someone pipettes. The camera code (`scripts/live_demo.py --ui`)
reports every dispense it has confirmed. The server compares it with the plan, and a wrong
well flashes and beeps on every open screen.

```
┌───────────────┐  edits the plan   ┌──────────────────────┐  GET /api/plan (A1 corner)  ┌─────────────────────┐
│ browser (UI)  │ ────────────────> │ app.py (this server) │ <────────────────────────── │ live_demo.py --ui   │
│ plan, run,    │ <──────────────── │ plan + SQLite file   │ <────────────────────────── │ camera + tip model  │
│ alarm, log    │   live updates    └──────────────────────┘  POST /api/drop {"wells"}   └─────────────────────┘
```

Built with Starlette (web framework), uvicorn (server), SQLite (comes with Python), and plain
HTML, CSS and JavaScript. No build step. Python 3.11 or newer.

## 1. Run it

```
pip install -r requirements.txt
python app.py
```

Open http://localhost:7500. Your data lives in `data/wellPlate.db` (copy it to back it up,
delete it to start fresh). Options: `python app.py --port 8080 --db D:\labData\plates.db`.

From a tablet on the same Wi-Fi: `python app.py --host 0.0.0.0`, then open the address it
prints. Every open screen stays in sync.

**As an app, full screen (Mac):**
- `python app.py --app` opens the UI in its own Chrome (or Edge) window with no tabs or address bar, full screen.
- Add `--windowed` for a normal-sized window instead.
- In the page, the full screen button at the top right (or the **F** key) switches full screen on and off. **Esc**
  also leaves it.
- To get a Dock icon: in Chrome, use the install icon in the address bar (or ⋮ → Cast, save and share → Install
  page as app). In Safari, use File → Add to Dock. PipetteCoach then opens like any other Mac app. The server
  (`python app.py`) still has to be running.

## 2. Use it

- **Left panel:** search and the substance list. Click a substance to select it. The form below
  adds a substance (preset color, **Custom** color picker, or a hex code like `#7A3E00`). The **⋯**
  button on each substance edits or deletes it.
- **Plan tab:** the plan is a list of **steps** (shown in the **Protocol** panel on the right):
  - Select wells (tap, drag, or click a row letter / column number), pick the substance on the
    left, then press **Confirm**: that substance is added to those wells as the next step.
  - A well can take several substances. They are added in step order, and the well shows one
    colored band per substance, top to bottom. Hover a well to see the list.
  - **Add timer** puts a wait (e.g. 1 min incubation) after the last step. Timers run **per well**:
    a well's timer starts the moment the substance before it goes into that well, and only that well
    waits for its next substance (e.g. A into A1, B1, C1, wait 1 min, then B: A1 can take B one
    minute after its own A went in, even while B1 and C1 are still timing).
  - **Erase** + Confirm clears the selected wells. The × on a step removes that step.
  - **Inspect** (or right-click a well with any tool) shows what goes into that well, in order.
    For a well with 2+ substances, untick **Must follow this order** to let them go in in any
    order (the well gets a ⇄ badge). The button under it sets the same for every well.
- **Run tab:**
  - **Guided / Free** (top left of the strip) picks how the app guides you. Every drop is checked
    the same way in both.
    - **Guided:** one step at a time, with one next well.
    - **Free:** hold any substance and tap it on the left. Every well that can take it now lights
      up, and each substance shows "N ready". Each well still follows its own sequence, so if well 1
      gets A then B and well 2 gets B then A, holding B lights only well 2. After that, A lights
      both. Any-order wells take their substances in any order, and only a timer that is already
      running holds them.
  - The top strip shows the run state and step, the substance **in the tip**, the **next well**
    that is ready (also outlined on the plate, column by column: A1, B1 … H1, A2 …) or, when
    every well left is still timing, the **timer** that ends first, and progress.
  - A well on its timer has an orange ring on its edge that empties as the time runs out, and
    the time left in the middle. A chime and a message say when a well is ready.
  - Tap a well to see what goes in it and what is already added.
  - **Next well** overrides the camera: the outlined well counts as dispensed with the planned
    substance, exactly like a detected drop, and the outline moves to the next well. The log
    shows it as a Match with source "Manual". It is hidden while a timer runs.
  - **Record each run** (in **Camera & test**): from **Start run** until the run ends or is reset, the
    camera saves the **raw** camera video (no drawings, so it can be annotated later) to
    `runs/<live demo>/run_<time>.mp4`, and next to it `run_<time>.json`: the dispenses it reported, each with
    `t_video` (seconds into the video). Each run gets its own file. The camera window shows a REC mark. A blinking **REC** next to the run state
    means the camera confirmed it is recording. It needs the camera to be running (**Start camera**).
  - Once per setup, open **Camera & test** (bottom right) and set **Well A1 in the camera
    picture**: the corner where A1 is, as the camera sees the plate. The camera code uses it to
    name the wells; if it is wrong, every well is wrong. It shows "A1 corner not set" in amber
    until you do.
  - Press **Start run**, then pipette. Tap the substance that is in the tip on the left (the
    camera sees where the tip goes, not what it holds).
  - A mistake flashes red, shows a banner and beeps: a wrong substance, an unplanned well, a
    second drop, a substance out of order in a well, or a substance added **before its timer
    finished**. Press **Acknowledge** to clear it. A finished timer chimes.
- **No camera?** Use *Correct drop* / *Wrong drop* under **Camera & test**, or run
  `python simulateCamera.py` (walks the steps and timers, sometimes into a wrong well).
- **Live tip** (Run tab): while the camera (or the Camera simulator) sees the tip, the well under it
  gets a ring and the bar under the top strip says what a drop there would be, for the substance
  selected on the left: **green** + a soft ding = right well; **red** = wrong well (with why: nothing
  planned, needs another substance, already filled, wrong order, timer running). Staying 0.7 s over a
  wrong well = about to dispense: the bar flashes and a warning sounds, before anything is dropped.
- **Camera simulator** (link at the top right, or http://localhost:7500/camera): a fake camera in
  another tab. Put both tabs side by side. Mouse over a well = the tip is over it (the Run tab rings
  it green or red); click a well to hold the tip there; **Dispense** (or D / Enter) sends the drop,
  judged like a camera drop. Its drops show as source "Simulator" in Records.
- **Records tab:** the detection log for the current run, and the plan the camera reads.

Editing wells during a run resets the run (the page tells you).

## 3. Connect the camera

**From the page:** under **Camera & test** press **Find cameras** (a snapshot of each camera; click the one that
sees the plate, or type a number / stream URL), then press **Start camera** on the Run tab. Camera mounted mirrored or upside down: **Flip the picture**
(left-right / up-down; the camera code picks it up within a few seconds, kept in `data/cameraFlip.json`). Then click
the corner of the small plate where well A1 is in the camera window (set the flip first: flipping moves A1). The server starts `scripts/live_demo.py --camera <n> --show --ui <this server>` on this computer, with
its own window showing the camera picture; **Stop** ends it. Only works from the computer running the server.
Its output goes to `data/liveCamera.log`, and the page says why if it stops by itself.

**By hand** (same thing, two terminals):

```
python app.py                                                       # terminal 1
python scripts/live_demo.py --camera 0 --show --ui http://localhost:7500   # terminal 2, repo root
```

`live_demo.py --ui` sends one `POST /api/drop {"wells": ["B4"]}` for each dispense whose
liquid check says "liquid". A stop with no liquid seen, or one the check cannot judge, is not
sent, so a short pause over a neighbouring well never raises an alarm. All calls run in a
background thread, so a slow server never slows the camera. It also sends a heartbeat every
3 s (the page shows **Camera: online**) and re-reads the A1 corner.

### The plan the camera reads (`GET /api/plan`)

```json
{
  "plate": {"wells": 96, "rows": 8, "columns": 12},
  "a1Corner": "bottomLeft",
  "status": "active",
  "activeCompoundId": 1,
  "compounds": [{"id": 1, "name": "IL-6 standard", "tag": "STD", "volumeUl": 100.0, "color": "#2F6FDE"}],
  "steps": [{"kind": "add", "compoundId": 1, "wells": ["A1", "A2"]},
            {"kind": "wait", "seconds": 300},
            {"kind": "add", "compoundId": 2, "wells": ["A1", "B1"]}],
  "wells": {"A1": [1, 2], "A2": [1], "B1": [2]}
}
```

`steps` is the protocol in order; `wells` lists each well's substances in the order they go in.

`a1Corner` is `auto`, `topLeft`, `topRight`, `bottomLeft` or `bottomRight`.

### `POST /api/drop`

```json
{"wells": ["B4"]}
→ {"accepted": true, "results": [{"well": "B4", "kind": "ok", "planned": "IL-6 standard"}], "alarm": null, "complete": false}
```

`kind` is `ok` (the substance this well needs next), `wrong` (the well needs another substance),
`unplanned` (nothing planned there), `duplicate` (already added), `order` (planned here, but an
earlier substance is not in the well yet) or `early` (a timer before this step has not finished). Optional `"compoundId": 3`
overrides the substance selected on the page. Before Start run, or with no substance selected,
the answer is HTTP 200 with `"accepted": false` and a `reason`. A well that is not on the plate
(`A13`) is HTTP 400, which usually means the camera calibration is off.

### All endpoints

| Method and path | Used by | Notes |
|---|---|---|
| `POST /api/drop` | camera | token required if one is set |
| `POST /api/camera/heartbeat` | camera | token required if one is set |
| `POST /api/camera/tip` | camera | `{"well": "B4"}` or `{"well": null}`, when it changes; token required if one is set |
| `GET /api/plan` | camera, Records tab | the plan above |
| `GET /api/state` | browser | everything the page draws |
| `GET /api/stream` | browser | live updates (Server-Sent Events) |
| `GET /api/health` | anyone | `{"ok": true}` |
| `POST /api/compounds`, `PUT` or `DELETE /api/compounds/{id}` | browser | |
| `PUT /api/plan` | browser | `{"plan": [steps]}` |
| `POST /api/activeCompound`, `POST /api/a1Corner` | browser | `{compoundId}`, `{a1Corner}` |
| `POST /api/run/start`, `/api/run/reset`, `/api/run/nextWell`, `/api/alarm/ack` | browser | |
| `POST /api/testDrop` | browser | the Run tab's test buttons and well taps |
| `GET /camera`, `POST /api/sim/drop`, `/api/sim/tip`, `/api/sim/heartbeat` | camera simulator | judged like camera drops; no token |

Errors are always `{"error": "readable message"}` with status 400 (bad input), 401 (token),
or 404 (not found).

## 4. The token (when other devices can reach the port)

Without a token, anyone who can reach the port can send fake drops. Set the same secret on
both sides before starting (`export WELLPLATE_API_TOKEN="pick-a-long-secret"`; PowerShell:
`$env:WELLPLATE_API_TOKEN = "..."`). `live_demo.py` and `simulateCamera.py` read the same
variable. Keep it out of the code and out of chat. The page itself has no login.

## 5. Files

| File | What it does |
|---|---|
| `app.py` | the web server: routes, live updates, token check |
| `plateLogic.py` | the rules: well ids, validation, and the right-or-wrong check |
| `store.py` | the SQLite database (substances, plan, run state, log) |
| `static/index.html`, `style.css`, `app.js` | the page (light theme) |
| `simulateCamera.py` | a fake camera for testing without one |
| `tests/runTests.py` | tests: `python tests/runTests.py` |
| `../pipette_coach/ui_link.py` | the camera side: background sender used by `live_demo.py --ui` |

## 6. Known limits

- **96-well plates and one tip at a time only**, because that is what the camera tracker knows.
- **No login.** Fine on a private lab PC.
- **One plate and one run at a time.** The log is cleared by Start run and Reset run.
- **A drop is only reported when the liquid check sees liquid.** A dispense the check misses
  stays "not done" on the page; it never raises a false alarm, but it is not caught either.
- **Sound** only plays after you have clicked somewhere on the page (a browser rule).
- **Not tested here:** Safari, Firefox and a real tablet.
