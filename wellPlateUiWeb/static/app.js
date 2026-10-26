/* Well Plate UI: browser code.
 *
 * The server owns the data. This page edits the plan locally (so painting feels
 * instant), sends it to the server, and redraws whenever the server pushes a
 * new state (Server-Sent Events). Names are camelCase.
 */
(function () {
    "use strict";

    var $ = function (s) { return document.querySelector(s); };
    var ROWS = 8, COLS = 12;  // 96-well plates only: that is what the camera tracker knows
    // Pastel palette, but saturated enough to read clearly on the plate
    var PALETTE = ["#8EC5FF", "#FF9EB5", "#FFB57A", "#7EDDB0", "#B79CF2", "#FFE066", "#6FD6E0", "#B4BEC9"];
    var TITLES = { wrong: "Wrong substance", unplanned: "Unplanned well", duplicate: "Duplicate drop", order: "Out of order", early: "Too early" };
    var LABELS = { ok: "Match", wrong: "Wrong substance", unplanned: "Unplanned", duplicate: "Duplicate", order: "Out of order", early: "Too early" };
    var SOURCES = { camera: "Camera", test: "Test", manual: "Manual", simulator: "Simulator" };

    var S = {
        compounds: [],
        steps: [], seq: {}, a1Corner: "auto", mode: "guided", anyOrder: {},  // steps: [{kind: "add", compoundId, wells} | {kind: "wait", seconds}]
        run: { status: "draft", activeCompoundId: null, done: {}, alarm: null, progress: { done: 0, total: 0 }, timers: {}, serverTime: 0 },
        events: [],
        camera: { online: false, lastSeen: null },
        live: { running: false, camera: null, exitCode: null, log: [] },  // the camera process started from this page
        activeId: null, editingId: null, view: "plan", tool: "paint", query: "", staged: {}, hoverStep: null,
        clockOffset: 0,
        connected: false, loaded: false, serverId: null, revision: -1
    };
    var drag = null, cellEls = [], toastTimer = null;
    var saving = false, pendingSave = false, activePending = 0;
    var lastServerActive = null, lastAlarmKey = null, payloadText = "", eventSource = null;

    /* ---------- helpers ---------- */
    function byId(id) {
        for (var i = 0; i < S.compounds.length; i++) if (S.compounds[i].id === id) return S.compounds[i];
        return null;
    }
    function wid(r, c) { return String.fromCharCode(65 + r) + (c + 1); }
    function parseId(id) { return { r: id.charCodeAt(0) - 65, c: parseInt(id.slice(1), 10) - 1 }; }
    function rgb(h) {
        h = h.replace("#", "");
        return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
    }
    function textOn(hex) { var o = rgb(hex); return (0.299 * o.r + 0.587 * o.g + 0.114 * o.b) > 165 ? "#10181d" : "#ffffff"; }
    function el(tag, cls, text) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text != null) e.textContent = text;
        return e;
    }
    function fmtVol(v) { return (Math.round(v * 10) / 10) + " µL"; }
    function clock(iso) { return iso ? new Date(iso).toLocaleTimeString() : ""; }
    function colMajor(a, b) { var p = parseId(a), q = parseId(b); return p.c - q.c || p.r - q.r; }
    function fmtTime(sec) {
        sec = Math.max(0, Math.round(sec));
        var h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), x = sec % 60;
        return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(x).padStart(2, "0");
    }
    function fmtWait(sec) {
        if (sec % 3600 === 0) return sec / 3600 + " h";
        if (sec % 60 === 0) return sec / 60 + " min";
        return sec < 60 ? sec + " s" : fmtTime(sec);
    }

    /* ---------- the plan: an ordered list of steps ---------- */
    function computeSeq() {  // {well: [step index, ...]}: what goes into each well, in order
        S.seq = {};
        S.steps.forEach(function (st, i) {
            if (st.kind === "add") st.wells.forEach(function (w) { (S.seq[w] = S.seq[w] || []).push(i); });
        });
    }
    function running() { return S.run.status !== "draft"; }
    function stepDone(i, w) { var d = S.run.done[String(i)]; return running() && !!(d && d[w]); }
    function stepComplete(i) {
        var st = S.steps[i];
        return st.kind === "add" && st.wells.every(function (w) { return stepDone(i, w); });
    }
    function serverNow() { return Date.now() / 1000 + S.clockOffset; }
    /* Timers run per well (server: plateLogic.wellTimers): a well's timer starts when its compound goes in and
       holds only that well's next compound. S.run.wellTimers = {well: [{step, seconds, endsAt}]}. */
    function wellTimerList(w) { return (S.run.wellTimers || {})[w] || []; }
    function liveTimer(t) {  // {state: pending|running|done, left: seconds}
        if (t.endsAt == null) return { state: "pending", left: t.seconds };
        var left = t.endsAt - serverNow();
        return { state: left > 0 ? "running" : "done", left: Math.max(0, left) };
    }
    function runningTimerOf(w) {  // the well's running timer {step, seconds, left}, or null
        var list = wellTimerList(w);
        for (var k = 0; k < list.length; k++) {
            var x = liveTimer(list[k]);
            if (x.state === "running") return { step: list[k].step, seconds: list[k].seconds, left: x.left };
        }
        return null;
    }
    function timerOf(i) {  // wait step i over its wells: {state, running: wells timing, left: the soonest end}
        var n = 0, run = 0, done = 0, left = null, all = S.run.wellTimers || {};
        Object.keys(all).forEach(function (w) {
            all[w].forEach(function (t) {
                if (t.step !== i) return;
                n++;
                var x = liveTimer(t);
                if (x.state === "running") { run++; left = left == null ? x.left : Math.min(left, x.left); }
                if (x.state === "done") done++;
            });
        });
        var state = !running() ? "pending" : !n || done === n ? "done" : run ? "running" : "pending";
        return { state: state, running: run, left: left };
    }
    function stepFinished(i) { return S.steps[i].kind === "add" ? stepComplete(i) : timerOf(i).state === "done"; }
    function currentStep() {  // the first compound step not finished (timers do not block the whole plate)
        if (!running()) return null;
        for (var i = 0; i < S.steps.length; i++) if (S.steps[i].kind === "add" && !stepComplete(i)) return i;
        return null;
    }
    function anyOrder(w) { return !!S.anyOrder[w]; }  // this well takes its substances in any order
    function wellReady(i, w) {  // step i's compound can go into w now: earlier compounds in, its timer done
        if (stepDone(i, w)) return false;
        var seq = S.seq[w] || [], free = anyOrder(w);
        if (!free) for (var k = 0; k < seq.length && seq[k] < i; k++) if (!stepDone(seq[k], w)) return false;
        // any order: only a RUNNING timer holds the well (one whose substance is not in yet has not started)
        return !wellTimerList(w).some(function (t) {
            var st = liveTimer(t).state;
            return t.step < i && (free ? st === "running" : st !== "done");
        });
    }
    function freeRun() { return S.mode === "free"; }
    function readyFor(cid) {  // Free run: every well that can take this substance now (its own sequence decides)
        if (!running() || cid == null) return [];
        return Object.keys(S.seq).filter(function (w) {
            return S.seq[w].some(function (i) { return S.steps[i].compoundId === cid && wellReady(i, w); });
        }).sort(colMajor);
    }
    function heldFor(cid) {  // Free run: wells that need this substance but wait on a timer, the one ending first
        var best = null;
        if (!running() || cid == null) return null;
        Object.keys(S.seq).forEach(function (w) {
            var t = runningTimerOf(w);
            if (!t) return;
            var wants = S.seq[w].some(function (i) { return S.steps[i].compoundId === cid && !stepDone(i, w) && i > t.step; });
            if (wants && (!best || t.left < best.left)) best = { well: w, left: t.left };
        });
        return best;
    }
    function nextWellOfStep(i) {
        if (i == null || S.steps[i].kind !== "add") return null;
        var w = S.steps[i].wells.filter(function (w) { return wellReady(i, w); }).sort(colMajor);
        return w[0] || null;
    }
    function waitingTimer(i) {  // step i's wells still held by a timer: the one that ends first {well, left}
        var best = null;
        if (i == null) return null;
        S.steps[i].wells.forEach(function (w) {
            var t = stepDone(i, w) ? null : runningTimerOf(w);
            if (t && t.step < i && (!best || t.left < best.left)) best = { well: w, left: t.left };
        });
        return best;
    }
    function compoundCounts() {  // {compoundId: [done, total]} counting each compound in each well
        var out = {};
        S.steps.forEach(function (st, i) {
            if (st.kind !== "add") return;
            var c = out[st.compoundId] = out[st.compoundId] || [0, 0];
            st.wells.forEach(function (w) { c[1]++; if (stepDone(i, w)) c[0]++; });
        });
        return out;
    }
    function stepText(st) {
        if (st.kind === "wait") return "Wait " + fmtWait(st.seconds);
        var c = byId(st.compoundId);
        return (c ? c.name : "?") + " \u00B7 " + st.wells.length + (st.wells.length === 1 ? " well" : " wells");
    }

    function toast(msg, actLabel, actFn) {
        var t = $("#toast");
        t.replaceChildren(el("span", null, msg));
        if (actLabel) {
            var b = el("button", null, actLabel);
            b.onclick = function () { actFn(); t.hidden = true; };
            t.appendChild(b);
        }
        t.hidden = false;
        clearTimeout(toastTimer);
        toastTimer = setTimeout(function () { t.hidden = true; }, actLabel ? 6000 : 2800);
    }

    /* ---------- talking to the server ---------- */
    function api(method, url, body) {
        var opts = { method: method, headers: {} };
        if (body !== undefined) {
            opts.headers["Content-Type"] = "application/json";
            opts.body = JSON.stringify(body);
        }
        return fetch(url, opts).then(function (res) {
            return res.json().catch(function () { return {}; }).then(function (data) {
                if (!res.ok) throw new Error(data.error || ("Server error " + res.status));
                return data;
            });
        }, function () {
            throw new Error("Cannot reach the server. Is it still running?");
        });
    }
    function fail(err) { toast(err && err.message ? err.message : "Something went wrong."); }
    function reloadState() { api("GET", "/api/state").then(applyState).catch(function () {}); }

    function applyState(st, force) {
        // A slow answer to an earlier request must never overwrite a newer live
        // update (it could hide an alarm). Numbers restart when the server restarts.
        // `force` re-applies the same state (used once a plan save finishes, because
        // the plan part is skipped while a save is in flight).
        if (st.serverId === S.serverId && (st.revision < S.revision || (st.revision === S.revision && !force))) return;
        if (!st.plate || !Array.isArray(st.plate.steps)) {  // app.py still running an older version
            $("#serverDot").dataset.state = "warn";
            $("#serverText").textContent = "Server: restart needed";
            toast("The server is running an older version of the app. Stop python app.py (Ctrl+C) and start it again.");
            return;
        }
        S.serverId = st.serverId;
        S.revision = st.revision;
        var first = !S.loaded;
        S.loaded = true;
        S.compounds = st.compounds;
        var fresh = first ? [] : newlyDone(S.run, st.run), wasComplete = S.run.status === "complete";
        S.run = st.run;
        S.events = st.events;
        S.camera = st.camera || S.camera;
        S.live = st.live || S.live;
        S.cameraFlip = st.cameraFlip || S.cameraFlip;
        if (st.tip) setTip(st.tip, true);
        S.a1Corner = st.plate.a1Corner;
        S.mode = st.run.mode || "guided";
        S.anyOrder = {};
        (st.plate.anyOrder || []).forEach(function (w) { S.anyOrder[w] = true; });
        S.clockOffset = (st.run.serverTime || Date.now() / 1000) - Date.now() / 1000;
        $("#cornerSelect").value = S.a1Corner;
        $("#recordBox").checked = !!st.run.record;

        // The plan is edited locally. While a save is in flight, do not let an
        // older server copy overwrite what the user just painted.
        if (!(saving || pendingSave)) { S.steps = st.plate.steps; computeSeq(); }

        // Which compound is selected. The server keeps a copy: the camera sees
        // where the tip goes, not what it holds.
        var serverActive = st.run.activeCompoundId;
        var needPost = false;
        if (first) {
            S.activeId = serverActive != null && byId(serverActive) ? serverActive : (S.compounds.length ? S.compounds[0].id : null);
            needPost = S.activeId !== serverActive;
        } else if (S.activeId != null && !byId(S.activeId)) {
            S.activeId = S.compounds.length ? S.compounds[0].id : null;
            needPost = true;
        } else if (serverActive !== lastServerActive && activePending === 0 && serverActive != null && byId(serverActive)) {
            S.activeId = serverActive;  // someone else changed it
        }
        lastServerActive = serverActive;
        if (needPost) postActive();
        if (S.editingId != null && !byId(S.editingId)) stopEditing();  // deleted on another screen

        var alarm = st.run.alarm;
        var key = alarm ? alarm.time + ":" + alarm.count : null;
        if (key && key !== lastAlarmKey) beep();
        lastAlarmKey = key;
        document.title = alarm ? "ALARM: " + TITLES[alarm.kind] + " - Well Plate UI" : "Well Plate UI";

        if (S.view === "records") loadPayload();
        renderAll();
        if (fresh.length) {  // a dispense was counted: sound + a green pop on the well(s)
            if (st.run.status === "complete" && !wasComplete) completeSound(); else dispensedSound();
            popWells(fresh);
        }
    }
    function newlyDone(before, after) {  // wells that got a substance between two states of the same run
        if (!after || after.status === "draft" || before.startedAt !== after.startedAt) return [];
        var out = [];
        Object.keys(after.done || {}).forEach(function (i) {
            var was = (before.done || {})[i] || {};
            Object.keys(after.done[i]).forEach(function (w) { if (!was[w] && out.indexOf(w) < 0) out.push(w); });
        });
        return out;
    }
    function popWells(ids) {
        cellEls.forEach(function (w) {
            if (ids.indexOf(w.dataset.id) < 0) return;
            w.classList.remove("justDone");
            void w.offsetWidth;  // restart the animation if it is still running
            w.classList.add("justDone");
            setTimeout(function () { w.classList.remove("justDone"); }, 1300);
        });
    }

    function connect() {
        if (!window.EventSource) return;  // the 4-second poll below still works
        eventSource = new EventSource("/api/stream");
        eventSource.onopen = function () { S.connected = true; renderLink(); };
        eventSource.addEventListener("state", function (e) { S.connected = true; applyState(JSON.parse(e.data)); });
        eventSource.addEventListener("tip", function (e) { setTip(JSON.parse(e.data)); });
        eventSource.onerror = function () {
            S.connected = false;
            renderLink();
            if (eventSource.readyState === 2) {  // closed for good: open a new one
                eventSource.close();
                setTimeout(connect, 2000);
            }
        };
    }

    function savePlan() {
        if (saving) { pendingSave = true; return; }
        saving = true;
        pendingSave = false;
        api("PUT", "/api/plan", { plan: S.steps }).then(function (res) {
            saving = false;
            if (res.runReset) toast("The plan changed, so the run was reset.");
            if (pendingSave) savePlan(); else applyState(res.state, true);
        }).catch(function (err) {
            saving = false;
            pendingSave = false;
            fail(err);
            reloadState();
        });
    }

    function postActive() {
        activePending++;
        api("POST", "/api/activeCompound", { compoundId: S.activeId }).then(function (res) {
            applyState(res.state);
        }).catch(fail).then(function () { activePending--; });
    }

    /* ---------- compound library ---------- */
    function renderList() {
        var list = $("#compoundList");
        list.replaceChildren();
        if (!S.loaded) { list.appendChild(el("li", "emptyNote", "Loading…")); return; }
        var counts = compoundCounts();
        var q = S.query.trim().toLowerCase();
        var arr = S.compounds.filter(function (c) {
            return !q || c.name.toLowerCase().indexOf(q) > -1 || c.tag.toLowerCase().indexOf(q) > -1;
        }).sort(function (a, b) {  // A to Z, ignoring case; "Std 2" before "Std 10"
            return a.name.localeCompare(b.name, undefined, { sensitivity: "base", numeric: true });
        });
        if (!arr.length) {
            list.appendChild(el("li", "emptyNote", S.compounds.length ? "No substances match your search." : "No substances yet. Add one below."));
            return;
        }
        arr.forEach(function (c) {
            var li = el("li", "cardRow"), b = el("button", "card");
            b.type = "button";
            b.setAttribute("aria-pressed", c.id === S.activeId ? "true" : "false");
            var sw = el("span", "cardColor");
            sw.style.background = c.color;
            var n = (counts[c.id] || [0, 0])[1], d = (counts[c.id] || [0, 0])[0];
            b.title = c.name + (c.tag ? " (" + c.tag + ")" : "");
            b.appendChild(sw);
            b.appendChild(el("span", "cardName", c.name));
            b.appendChild(el("span", "cardCount", fmtVol(c.volumeUl) + " \u00B7 " + (S.view === "run" ? d + " of " + n + " dispensed" : n + (n === 1 ? " well" : " wells"))));
            var nReady = S.view === "run" && freeRun() ? readyFor(c.id).length : 0;
            if (nReady) b.appendChild(el("span", "readyBadge", nReady + " ready"));
            if (S.view === "run" && n) {
                var m = el("div", "cardMini"), i = el("i");
                i.style.width = (100 * d / n) + "%";
                m.appendChild(i);
                b.appendChild(m);
            }
            b.onclick = function () { S.activeId = c.id; postActive(); renderAll(); };
            li.appendChild(b);
            var dots = el("button", "dots", "\u22EF");
            dots.type = "button";
            dots.setAttribute("aria-label", "Options for " + c.name);
            dots.setAttribute("aria-haspopup", "menu");
            dots.setAttribute("aria-expanded", String(menuFor === c.id));
            dots.onclick = function (e) { e.stopPropagation(); openMenu(c.id, dots); };
            li.appendChild(dots);
            list.appendChild(li);
        });
    }
    function loadEditor(c) {
        $("#nameField").value = c.name;
        $("#tagField").value = c.tag;
        $("#volField").value = c.volumeUl;
        setColor(c.color);
    }
    function renderEditorState() {
        var c = byId(S.editingId);
        $("#editorTitle").textContent = c ? "Edit " + c.name : "New substance";
        $("#saveBtn").textContent = c ? "Save" : "Add";
    }
    function clearEditor() {
        $("#nameField").value = "";
        $("#tagField").value = "";
        $("#volField").value = 8;
        setColor(nextColor());
    }
    // The add/edit form lives in a dialog, so the compound list gets the whole left column
    function openEditor() {
        var d = $("#compoundDialog");
        if (!d.open) d.showModal();
        $("#nameField").focus();
    }
    function startEditing(c) {
        S.editingId = c.id;
        loadEditor(c);
        renderEditorState();
        openEditor();
    }
    function startAdding() {
        S.editingId = null;
        clearEditor();
        renderEditorState();
        openEditor();
    }
    function stopEditing() {
        S.editingId = null;
        clearEditor();
        renderEditorState();
        if ($("#compoundDialog").open) $("#compoundDialog").close();
    }
    $("#newCompoundBtn").onclick = startAdding;
    $("#compoundDialog").addEventListener("close", function () { if (S.editingId != null) stopEditing(); });
    $("#editor").addEventListener("keydown", function (e) { if (e.key === "Enter" && e.target.tagName === "INPUT") $("#saveBtn").click(); });

    /* ---------- the "..." menu on each compound ---------- */
    var menuFor = null;
    function openMenu(cid, anchor) {
        if (menuFor === cid) { closeMenu(); return; }
        menuFor = cid;
        var m = $("#cardMenu"), r = anchor.getBoundingClientRect();
        m.hidden = false;
        m.style.top = Math.min(r.bottom + 4, window.innerHeight - m.offsetHeight - 8) + "px";
        m.style.left = Math.max(8, r.right - m.offsetWidth) + "px";
        renderList();
        m.querySelector("button").focus();
    }
    function closeMenu() {
        if (menuFor == null) return;
        menuFor = null;
        $("#cardMenu").hidden = true;
        renderList();
    }
    $("#cardMenu").onclick = function (e) {
        var b = e.target.closest("button"), c = byId(menuFor);
        closeMenu();
        if (!b || !c) return;
        if (b.dataset.act === "edit") startEditing(c);
        else deleteCompound(c);
    };
    document.addEventListener("click", function (e) { if (!e.target.closest("#cardMenu")) closeMenu(); });
    document.addEventListener("keydown", function (e) {
        if (e.key !== "Escape") return;
        if (menuFor != null) closeMenu(); else if (S.editingId != null) stopEditing();
    });
    // Full screen: the button or F (not while typing). Esc leaves (the browser does that itself).
    function isFull() { return !!(document.fullscreenElement || document.webkitFullscreenElement); }
    function toggleFull() {
        var d = document.documentElement;
        if (isFull()) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
        else {
            var p = (d.requestFullscreen || d.webkitRequestFullscreen).call(d);
            if (p && p.catch) p.catch(function () { toast("Full screen was blocked by the browser."); });
        }
    }
    function showFull() {
        document.body.classList.toggle("isFull", isFull());
        $("#fullBtn").setAttribute("aria-pressed", String(isFull()));
        $("#fullBtn").title = isFull() ? "Leave full screen (F or Esc)" : "Full screen (F)";
    }
    $("#fullBtn").onclick = toggleFull;
    document.addEventListener("fullscreenchange", showFull);
    document.addEventListener("webkitfullscreenchange", showFull);
    document.addEventListener("keydown", function (e) {
        if ((e.key !== "f" && e.key !== "F") || e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.target.closest("input, textarea, select, [contenteditable]")) return;
        e.preventDefault();
        toggleFull();
    });
    $("#compoundList").addEventListener("scroll", closeMenu);
    window.addEventListener("resize", closeMenu);
    function readEditor() {
        var name = $("#nameField").value.trim();
        if (!name) { toast("Enter a substance name first."); $("#nameField").focus(); return null; }
        var vol = parseFloat($("#volField").value);
        if (isNaN(vol) || vol < 0) { toast("Volume must be 0 µL or more."); return null; }
        if ($("#hexField").classList.contains("bad")) { toast("Color must look like #2F6FDE."); $("#hexField").focus(); return null; }
        return { name: name, tag: $("#tagField").value.trim(), volumeUl: vol, color: $("#colorField").value };
    }
    function nextColor() {
        var used = S.compounds.map(function (c) { return c.color.toLowerCase(); });
        for (var i = 0; i < PALETTE.length; i++) if (used.indexOf(PALETTE[i].toLowerCase()) < 0) return PALETTE[i];
        return PALETTE[S.compounds.length % PALETTE.length];
    }

    $("#saveBtn").onclick = function () {
        var v = readEditor(), c = byId(S.editingId);
        if (!v) return;
        if (c) {
            api("PUT", "/api/compounds/" + c.id, v).then(function (res) {
                applyState(res.state);
                stopEditing();
                toast("Saved " + res.compound.name + ".");
            }).catch(fail);
            return;
        }
        api("POST", "/api/compounds", v).then(function (res) {
            applyState(res.state);
            S.activeId = res.compound.id;
            postActive();
            stopEditing();
            renderAll();
            toast("Added " + res.compound.name + ".");
            var sel = $('#compoundList [aria-pressed="true"]');  // the list is A-Z: show the new one wherever it landed
            if (sel) sel.scrollIntoView({ block: "nearest" });
        }).catch(fail);
    };
    $("#cancelBtn").onclick = stopEditing;
    function deleteCompound(c) {
        if (!window.confirm("Delete " + c.name + " and its steps in the plan?")) return;
        api("DELETE", "/api/compounds/" + c.id).then(function (res) {
            if (S.editingId === c.id) stopEditing();
            applyState(res.state);
            var n = res.deleted.wells;
            toast("Deleted " + c.name + (n ? " and its " + n + " wells." : ".") + (res.deleted.runReset ? " The run was reset." : ""));
        }).catch(fail);
    }
    $("#searchBox").oninput = function (e) { S.query = e.target.value; renderList(); };
    // Color: a preset swatch, any color from the picker, or a typed hex code
    function setColor(hex) {
        hex = hex.toLowerCase();
        $("#colorField").value = hex;
        $("#hexField").value = hex.toUpperCase();
        $("#hexField").classList.remove("bad");
        [].forEach.call($("#swatches").children, function (b) { b.setAttribute("aria-pressed", String(b.dataset.color === hex)); });
        $(".customColor").classList.toggle("on", PALETTE.map(function (p) { return p.toLowerCase(); }).indexOf(hex) < 0);
    }
    (function () {
        var sw = $("#swatches");
        PALETTE.forEach(function (p) {
            var b = el("button");
            b.type = "button";
            b.style.background = p;
            b.dataset.color = p.toLowerCase();
            b.setAttribute("aria-label", "Use color " + p);
            b.onclick = function () { setColor(p); };
            sw.appendChild(b);
        });
        $("#colorField").oninput = function (e) { setColor(e.target.value); };
        $("#hexField").oninput = function (e) {
            var v = e.target.value.trim();
            if (v && v[0] !== "#") v = "#" + v;
            var ok = /^#[0-9a-fA-F]{6}$/.test(v);
            e.target.classList.toggle("bad", !ok);
            if (ok) { $("#colorField").value = v.toLowerCase(); setColorMarks(v.toLowerCase()); }
        };
        function setColorMarks(hex) {
            [].forEach.call(sw.children, function (b) { b.setAttribute("aria-pressed", String(b.dataset.color === hex)); });
            $(".customColor").classList.toggle("on", PALETTE.map(function (p) { return p.toLowerCase(); }).indexOf(hex) < 0);
        }
    })();

    /* ---------- plate grid ---------- */
    var plate = $("#plate"), wrap = $("#plateWrap");
    function rebuildGrid() {
        var R = ROWS, C = COLS;
        plate.replaceChildren();
        cellEls = [];
        plate.style.gridTemplateColumns = "var(--hd) repeat(" + C + ", var(--cell))";
        plate.style.gridTemplateRows = "var(--hd) repeat(" + R + ", var(--cell))";
        plate.appendChild(el("div"));
        for (var c = 0; c < C; c++) {
            var h = el("div", "axisHead", String(c + 1));
            h.dataset.kind = "col";
            h.dataset.i = c;
            plate.appendChild(h);
        }
        for (var r = 0; r < R; r++) {
            var rh = el("div", "axisHead", String.fromCharCode(65 + r));
            rh.dataset.kind = "row";
            rh.dataset.i = r;
            plate.appendChild(rh);
            for (c = 0; c < C; c++) {
                var w = el("div", "well");
                w.dataset.r = r;
                w.dataset.c = c;
                w.dataset.id = wid(r, c);
                plate.appendChild(w);
                cellEls.push(w);
            }
        }
        fitGrid();
        updateCells();
    }
    function fitGrid() {
        var R = ROWS, C = COLS, pad = 8, gap = 3, hd = 20;  // keep in sync with .plate in style.css
        // Plan: the selection bar right under the plate (its space is kept while idle).
        // Run: the run strip sits above the plate area and the run buttons right under the plate (+ the 8px gap).
        var strip = S.view === "run" ? $(".runBar").offsetHeight : 0;
        var buttons = S.view === "run" ? $("#runActions").offsetHeight + 8 : 0;
        var bar = S.view === "plan" ? $("#stageBar").offsetHeight + 8 : 0;  // the selection bar under the plate
        var reserve = S.view === "plan" ? bar : S.view === "run" ? Math.max(buttons, 92 - strip) : 0;
        var W = wrap.clientWidth - 8, H = wrap.clientHeight - 8 - reserve;  // .plateWrap padding 4px
        // Narrow windows scroll (see style.css), so there the plate is sized by the width only
        if (window.matchMedia("(max-width: 760px)").matches) H = Infinity;
        var cell = Math.floor(Math.min((W - 2 * pad - hd - gap * C) / C, (H - 2 * pad - hd - gap * R) / R));
        cell = Math.max(12, Math.min(120, cell));  // big plates are easier to read from the bench
        plate.style.setProperty("--cell", cell + "px");
        plate.classList.toggle("showLabels", cell >= 22);
    }
    function dragWells() {
        if (!drag) return [];
        var out = [];
        var r1 = drag.r, c1 = drag.c;
        if (S.view === "run") { r1 = drag.r0; c1 = drag.c0; }
        var ra = Math.min(drag.r0, r1), rb = Math.max(drag.r0, r1), ca = Math.min(drag.c0, c1), cb = Math.max(drag.c0, c1);
        for (var r = ra; r <= rb; r++) for (var c = ca; c <= cb; c++) out.push(wid(r, c));
        return out;
    }
    var STRIPES = "repeating-linear-gradient(45deg, rgba(255, 255, 255, 0.8) 0 3px, transparent 3px 7px)";  // "still to add" hatching
    function alpha(hex, a) { var o = rgb(hex); return "rgba(" + o.r + "," + o.g + "," + o.b + "," + a + ")"; }
    // Sets p.pct on each part: its volume as a % of the well's total (equal shares if no volumes are set)
    function sharePct(parts) {
        var total = parts.reduce(function (s, p) { return s + Math.max(0, +p.c.volumeUl || 0); }, 0);
        parts.forEach(function (p) { p.pct = total > 0 ? 100 * Math.max(0, +p.c.volumeUl || 0) / total : 100 / parts.length; });
        return parts;
    }
    function fmtPct(p) { return (p >= 10 || p === 0 ? Math.round(p) : Math.round(p * 10) / 10) + "%"; }
    function updateCells() {
        var sel = {};
        dragWells().forEach(function (id) { sel[id] = 1; });
        var alarmSet = {};
        if (S.run.alarm) S.run.alarm.wells.forEach(function (id) { alarmSet[id] = 1; });
        var showLbl = plate.classList.contains("showLabels"), inRun = S.view === "run" && running();
        var cur = currentStep(), nextSet = {};
        if (S.view === "run" && !S.run.alarm) {  // guided: the one next well; free: every well ready for the tip
            if (freeRun()) readyFor(S.activeId).forEach(function (w) { nextSet[w] = 1; });
            else { var nw = nextWellOfStep(cur); if (nw) nextSet[nw] = 1; }
        }
        var hl = S.hoverStep != null && S.steps[S.hoverStep] && S.steps[S.hoverStep].kind === "add" ? S.steps[S.hoverStep].wells : [];
        var stagedC = byId(S.activeId);
        cellEls.forEach(function (w) {
            var id = w.dataset.id, seq = S.seq[id] || [];
            var parts = seq.map(function (i) {
                var c = byId(S.steps[i].compoundId) || { color: "#999999", tag: "?", name: "?", volumeUl: 0 };
                return { i: i, c: c, done: stepDone(i, id) };
            });
            var allDone = parts.length && parts.every(function (p) { return p.done; });
            sharePct(parts);
            if (parts.length) {
                // Pie chart: each compound's slice is its share of the well's total volume.
                // Run tab: an added slice is solid; a slice still to add is striped (white hatching over its color).
                // Three layers: solid added slices (top) / white stripes / every slice's color (bottom).
                var n = parts.length, acc = 0, solid = [], base = [];
                parts.forEach(function (p, k) {
                    var a = acc, b = k === n - 1 ? 100 : (acc += p.pct);
                    var range = " " + a + "% " + b + "%";
                    solid.push((inRun && !p.done ? "transparent" : p.c.color) + range);
                    base.push(p.c.color + range);
                });
                var anyPending = inRun && parts.some(function (p) { return !p.done; });
                w.style.background = !anyPending
                    ? (n === 1 ? parts[0].c.color : "conic-gradient(" + base.join(", ") + ")")
                    : "conic-gradient(" + solid.join(", ") + "), " + STRIPES + ", conic-gradient(" + base.join(", ") + ")";
                w.style.color = n === 1 && !(inRun && !parts[0].done) ? textOn(parts[0].c.color) : "";
                w.style.borderColor = "transparent";
            } else { w.style.background = ""; w.style.color = ""; w.style.borderColor = ""; }
            w.classList.toggle("planned", parts.length > 0);
            w.classList.toggle("multi", parts.length > 1);
            w.classList.toggle("pending", inRun && parts.length > 0 && !allDone);
            w.classList.toggle("done", !!(inRun && allDone));
            w.classList.toggle("alarm", !!alarmSet[id]);
            w.classList.toggle("sel", !!sel[id]);
            w.classList.toggle("staged", S.view === "plan" && !!S.staged[id]);
            w.classList.toggle("erase", S.view === "plan" && !!S.staged[id] && S.tool === "erase");
            w.classList.toggle("next", !!nextSet[id]);
            w.classList.toggle("hl", hl.indexOf(id) > -1);
            w.style.setProperty("--stage", S.tool === "paint" && stagedC ? stagedC.color : "var(--alarm)");
            w.textContent = "";
            var tm = inRun ? runningTimerOf(id) : null;
            // No text in wells: colors only, named in the compound list (hover a well for names and %)
            w.classList.toggle("timing", !!tm);
            if (tm) {  // countdown ring on the well's edge + the time left
                var tr = el("i", "timerRing");
                tr.style.setProperty("--p", tm.left / tm.seconds);
                w.appendChild(tr);
                w.appendChild(el("span", "timerLeft", fmtTime(tm.left)));
            }
            if (parts.length > 1 && anyOrder(id)) w.appendChild(el("i", "orderBadge", "\u21C4"));  // any order
            var tipHere = S.view === "run" && S.tip && S.tip.well === id;
            w.classList.toggle("tipHere", !!tipHere);
            if (tipHere) { var ring = el("i", "tipRing"); ring.dataset.state = tipState(); w.appendChild(ring); }
            w.title = id + (parts.length ? "\n" + parts.map(function (p, k) {
                return (anyOrder(id) ? "\u2022 " : (k + 1) + ". ") + p.c.name + " \u00B7 " + fmtVol(p.c.volumeUl) + " (" + fmtPct(p.pct) + ")" + (p.done ? "  \u2713" : "");
            }).join("\n") + (parts.length > 1 && anyOrder(id) ? "\nAny order" : "") : "  (empty)");
        });
        $("#plateShadow").classList.toggle("alarmed", !!S.run.alarm);
    }
    if (window.ResizeObserver) new ResizeObserver(function () { fitGrid(); updateCells(); }).observe(wrap);
    window.addEventListener("resize", function () { fitGrid(); updateCells(); });

    plate.addEventListener("pointerdown", function (e) {
        if (S.view === "records") return;
        var h = e.target.closest(".axisHead");
        if (h) {
            if (S.view !== "plan") return;
            var ids = [], i = +h.dataset.i;
            if (h.dataset.kind === "row") for (var c = 0; c < COLS; c++) ids.push(wid(i, c));
            else for (var r = 0; r < ROWS; r++) ids.push(wid(r, i));
            commit(ids, false);
            return;
        }
        if (e.button !== 0) return;  // right-click opens the well details (contextmenu below)
        var w = e.target.closest(".well");
        if (!w) return;
        e.preventDefault();
        try { plate.setPointerCapture(e.pointerId); } catch (x) {}
        drag = { r0: +w.dataset.r, c0: +w.dataset.c, r: +w.dataset.r, c: +w.dataset.c, moved: false };
        updateCells();
    });
    plate.addEventListener("pointermove", function (e) {
        if (!drag) return;
        var t = document.elementFromPoint(e.clientX, e.clientY), w = t && t.closest ? t.closest(".well") : null;
        if (w && (+w.dataset.r !== drag.r || +w.dataset.c !== drag.c)) {
            drag.r = +w.dataset.r;
            drag.c = +w.dataset.c;
            drag.moved = true;
            updateCells();
        }
    });
    plate.addEventListener("pointerup", function () {
        if (!drag) return;
        var ids = dragWells(), moved = drag.moved;
        drag = null;
        updateCells();
        if (ids.length) commit(ids, moved);
    });
    plate.addEventListener("pointercancel", function () { drag = null; updateCells(); });
    plate.addEventListener("contextmenu", function (e) {
        var w = e.target.closest(".well");
        if (!w || S.view === "records") return;
        e.preventDefault();
        showWellInfo(w.dataset.id);
    });

    function commit(ids, moved) {
        if (S.view === "run" || S.tool === "inspect") { if (ids.length === 1) showWellInfo(ids[0]); return; }
        if (S.view !== "plan") return;
        var all = ids.every(function (id) { return S.staged[id]; });
        ids.forEach(function (id) { if (all && !moved) delete S.staged[id]; else S.staged[id] = true; });
        renderAll();
    }
    function stagedWells() { return Object.keys(S.staged).sort(colMajor); }
    function clearStage() { S.staged = {}; renderAll(); }
    $("#stageCancel").onclick = clearStage;
    $("#stageConfirm").onclick = function () {
        var wells = stagedWells();
        if (!wells.length) return;
        if (S.tool === "erase") {
            S.steps.forEach(function (st) {
                if (st.kind === "add") st.wells = st.wells.filter(function (w) { return !S.staged[w]; });
            });
            S.steps = S.steps.filter(function (st) { return st.kind === "wait" || st.wells.length; });
            toast("Cleared " + wells.length + (wells.length === 1 ? " well." : " wells."));
        } else {
            var c = byId(S.activeId);
            if (!c) { toast("Select a substance first."); return; }
            var last = S.steps[S.steps.length - 1];
            if (last && last.kind === "add" && last.compoundId === c.id) {  // same compound again: grow that step
                wells.forEach(function (w) { if (last.wells.indexOf(w) < 0) last.wells.push(w); });
                last.wells.sort(colMajor);
            } else S.steps.push({ kind: "add", compoundId: c.id, wells: wells });
            toast("Added " + c.name + " to " + wells.length + (wells.length === 1 ? " well." : " wells."));
        }
        S.staged = {};
        computeSeq();
        savePlan();
        renderAll();
    };
    function renderStage() {
        // The bar always takes its space in the Plan tab (never hidden), so the plate does not
        // resize when wells are selected or confirmed. With nothing selected it shows a hint.
        var n = stagedWells().length, c = byId(S.activeId);
        $("#stageCancel").disabled = !n;
        $("#stageConfirm").disabled = !n;
        $("#stageBar").classList.toggle("idle", !n);
        var t = $("#stageText");
        t.replaceChildren();
        if (!n) t.appendChild(el("span", null, ""));
        else if (S.tool === "erase") t.appendChild(el("span", null, "Clear everything from " + n + (n === 1 ? " well" : " wells")));
        else if (c) {
            var tag = el("i", "dot");
            tag.style.background = c.color;
            t.appendChild(el("b", null, n + (n === 1 ? " well" : " wells")));
            t.appendChild(el("span", "muted", "\u2190"));
            t.appendChild(tag);
            t.appendChild(el("span", null, c.name));
            t.appendChild(el("span", "muted", "step " + stepNumberFor(c.id)));
        } else t.appendChild(el("span", null, "Pick a substance on the left"));
        $("#stageConfirm").textContent = S.tool === "erase" ? "Clear wells" : "Add to wells";
        $("#stageConfirm").className = "btn " + (S.tool === "erase" ? "dangerFill" : "primary");
    }
    function stepNumberFor(cid) {
        var last = S.steps[S.steps.length - 1];
        return last && last.kind === "add" && last.compoundId === cid ? S.steps.length : S.steps.length + 1;
    }

    /* ---------- protocol (step list) ---------- */
    function renderSteps() {
        var list = $("#stepList"), cur = currentStep(), inRun = S.view === "run" && running();
        list.replaceChildren();
        $("#stepsEmpty").hidden = S.steps.length > 0;
        S.steps.forEach(function (st, i) {
            var li = el("li", "step"), state = "";
            if (inRun) state = stepFinished(i) ? "done" : i === cur || (st.kind === "wait" && timerOf(i).state === "running") ? "current" : "";
            li.dataset.state = state;
            li.appendChild(el("span", "stepNo", state === "done" ? "\u2713" : String(i + 1)));
            var body = el("div", "stepBody");
            if (st.kind === "wait") {
                body.appendChild(el("span", "stepIcon", "\u23F1"));
                var t = timerOf(i), txt = "Wait " + fmtWait(st.seconds);
                if (inRun && t.state === "running") txt = "Wait " + fmtTime(t.left) + " (" + t.running + (t.running === 1 ? " well)" : " wells)");
                body.appendChild(el("span", "stepName", txt));
            } else {
                var c = byId(st.compoundId) || { tag: "?", color: "#999999", name: "?" };
                var tag = el("i", "dot");
                tag.style.background = c.color;
                body.appendChild(tag);
                body.appendChild(el("span", "stepName", c.name));
                var d = st.wells.filter(function (w) { return stepDone(i, w); }).length;
                body.appendChild(el("span", "stepMeta", inRun ? d + "/" + st.wells.length : st.wells.length + (st.wells.length === 1 ? " well" : " wells")));
            }
            li.appendChild(body);
            if (S.view === "plan") {
                var x = el("button", "stepDel", "\u00D7");
                x.type = "button";
                x.setAttribute("aria-label", "Remove step " + (i + 1));
                x.onclick = function () {
                    if (st.kind === "add" && !window.confirm("Remove step " + (i + 1) + " (" + stepText(st) + ")?")) return;
                    S.steps.splice(i, 1);
                    S.hoverStep = null;
                    computeSeq();
                    savePlan();
                    renderAll();
                };
                li.appendChild(x);
            }
            li.onmouseenter = function () { S.hoverStep = i; updateCells(); };
            li.onmouseleave = function () { S.hoverStep = null; updateCells(); };
            list.appendChild(li);
        });
    }

    /* ---------- timer dialog ---------- */
    $("#timerBtn").onclick = function () {
        if (!S.steps.length) { toast("Add a substance first, then the timer that follows it."); return; }
        $("#timerAfter").textContent = "Runs after step " + S.steps.length + " (" + stepText(S.steps[S.steps.length - 1]) + ").";
        $("#timerDialog").showModal();
        $("#timerMin").focus();
        $("#timerMin").select();
    };
    [].forEach.call(document.querySelectorAll("[data-preset]"), function (b) {
        b.onclick = function () { var v = +b.dataset.preset; $("#timerMin").value = Math.floor(v / 60); $("#timerSec").value = v % 60; };
    });
    $("#timerDialog").addEventListener("close", function () {
        if ($("#timerDialog").returnValue !== "add") return;
        var sec = (parseInt($("#timerMin").value, 10) || 0) * 60 + (parseInt($("#timerSec").value, 10) || 0);
        if (sec < 1 || sec > 86400) { toast("A timer must be between 1 second and 24 hours."); return; }
        S.steps.push({ kind: "wait", seconds: sec });
        computeSeq();
        savePlan();
        renderAll();
        toast("Timer added: wait " + fmtWait(sec) + ".");
    });

    /* ---------- well details (Run tab: tap a well; Plan tab: Inspect or right-click) ---------- */
    function showWellInfo(id) {
        var box = $("#wellInfo"), seq = S.seq[id] || [], cell = cellEls.filter(function (w) { return w.dataset.id === id; })[0];
        box.replaceChildren(el("b", "infoTitle", "Well " + id));
        if (!seq.length) box.appendChild(el("span", "muted", "Nothing planned."));
        var nextFound = false;
        var parts = sharePct(seq.map(function (i) { return { c: byId(S.steps[i].compoundId) || { volumeUl: 0 } }; }));
        seq.forEach(function (i, k) {
            var c = byId(S.steps[i].compoundId) || { tag: "?", color: "#999999", name: "?", volumeUl: 0 };
            var row = el("div", "infoRow"), tag = el("i", "dot"), done = stepDone(i, id);
            tag.style.background = c.color;
            row.appendChild(el("span", "muted", (k + 1) + "."));
            row.appendChild(tag);
            row.appendChild(el("span", null, c.name + " \u00B7 " + fmtVol(c.volumeUl) + " \u00B7 " + fmtPct(parts[k].pct)));
            var stt = done ? "\u2713 added" : anyOrder(id) ? (running() ? "to add" : "step " + (i + 1))
                : !nextFound && running() ? "next" : "step " + (i + 1);
            if (!done) nextFound = true;
            row.appendChild(el("span", "infoState " + (done ? "ok" : ""), stt));
            box.appendChild(row);
        });
        if (seq.length > 1) {  // the order option only means something with 2+ substances
            var lab = el("label", "infoOrder"), cb = document.createElement("input");
            cb.type = "checkbox";
            cb.checked = !anyOrder(id);
            lab.appendChild(cb);
            lab.appendChild(el("span", null, "Must follow this order"));
            cb.onchange = function () { setAnyOrder([id], !cb.checked); };
            box.appendChild(lab);
            var all = el("button", "btn small infoAll", cb.checked ? "Any order for every well" : "Order required for every well");
            all.type = "button";
            all.onclick = function () { setAnyOrder(Object.keys(S.seq), cb.checked); box.hidden = true; };
            box.appendChild(all);
        }
        var tm = running() ? runningTimerOf(id) : null;
        if (tm) box.appendChild(el("div", "infoRow infoTimer", "\u23F1 Timer: " + fmtTime(tm.left) + " left before step " + (tm.step + 2)));
        box.hidden = false;
        var r = cell.getBoundingClientRect();
        box.style.left = Math.min(r.right + 8, window.innerWidth - box.offsetWidth - 8) + "px";
        box.style.top = Math.min(r.top, window.innerHeight - box.offsetHeight - 8) + "px";
    }
    document.addEventListener("pointerdown", function (e) { if (!e.target.closest("#wellInfo, .well")) $("#wellInfo").hidden = true; });
    function setAnyOrder(wells, on) {
        api("POST", "/api/anyOrder", { wells: wells, anyOrder: on }).then(function (res) {
            applyState(res.state);
            var what = wells.length === 1 ? wells[0] : "Every well";
            toast(what + (on ? ": any order." : ": must follow its order."));
        }).catch(fail);
    }

    /* ---------- plan toolbar ---------- */
    $("#toolSeg").onclick = function (e) {
        var b = e.target.closest("button");
        if (!b) return;
        S.tool = b.dataset.tool;
        if (S.tool === "inspect") S.staged = {};
        $("#wellInfo").hidden = true;
        renderAll();
    };
    $("#clearBtn").onclick = function () {
        if (!S.steps.length || !window.confirm("Clear the whole plan (every step and timer)?")) return;
        S.steps = [];
        S.staged = {};
        computeSeq();
        savePlan();
        renderAll();
        toast("Plate cleared.");
    };
    /* ---------- plan file: export to share, import someone else's ----------
       The file names compounds instead of using database ids, so it works on any computer:
       {"format": "wellPlate.plan", "version": 1, "compounds": [{name, tag, volumeUl, color}],
        "steps": [{"kind": "add", "compound": 0, "wells": [...]}, {"kind": "wait", "seconds": 60}]}
       ("compound" is an index into "compounds"). */
    var PLAN_FORMAT = "wellPlate.plan";
    $("#exportBtn").onclick = function () {
        if (!S.steps.length) { toast("The plan is empty: add some wells first."); return; }
        var used = [], index = {};
        var steps = S.steps.map(function (st) {
            if (st.kind === "wait") return { kind: "wait", seconds: st.seconds };
            if (!(st.compoundId in index)) {
                var c = byId(st.compoundId) || { name: "Unknown", tag: "", volumeUl: 0, color: "#B4BEC9" };
                index[st.compoundId] = used.length;
                used.push({ name: c.name, tag: c.tag, volumeUl: c.volumeUl, color: c.color });
            }
            return { kind: "add", compound: index[st.compoundId], wells: st.wells.slice() };
        });
        var file = { format: PLAN_FORMAT, version: 1, exportedAt: new Date().toISOString(), compounds: used, steps: steps };
        var blob = new Blob([JSON.stringify(file, null, 2)], { type: "application/json" });
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "well-plate-plan-" + new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-") + ".json";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
        toast("Exported " + steps.length + (steps.length === 1 ? " step" : " steps") + " and " + used.length + (used.length === 1 ? " substance." : " substances."));
    };
    $("#importBtn").onclick = function () { $("#importFile").value = ""; $("#importFile").click(); };
    $("#importFile").onchange = function (e) {
        var f = e.target.files && e.target.files[0];
        if (!f) return;
        if (f.size > 2 * 1024 * 1024) { toast("That file is too big to be a plan."); return; }
        f.text().then(function (text) {
            var data;
            try { data = JSON.parse(text); } catch (x) { throw new Error("That file is not a plan (it is not valid JSON)."); }
            if (!data || data.format !== PLAN_FORMAT || !Array.isArray(data.compounds) || !Array.isArray(data.steps)) {
                throw new Error("That file is not a Well Plate plan. Use a file made with Export.");
            }
            if (data.version > 1) throw new Error("That plan was made by a newer version of this app.");
            var nAdd = data.steps.filter(function (st) { return st && st.kind === "add"; }).length;
            var msg = "Replace the current plan with \"" + f.name + "\"?\n\n" + data.steps.length + " steps, " +
                data.compounds.length + " substances. Substances you already have (same name) are reused; the others are added.";
            if (running()) msg += "\n\nThis resets the current run.";
            if (!nAdd || !window.confirm(msg)) { if (!nAdd) toast("That plan has no wells in it."); return; }
            return importPlan(data);
        }).catch(fail);
    };
    function importPlan(data) {
        // 1. map every compound in the file to one here: same name (any case) = same compound, else create it
        var ids = [], created = 0;
        var chain = Promise.resolve();
        data.compounds.forEach(function (fc, k) {
            chain = chain.then(function () {
                var name = String((fc && fc.name) || "").trim();
                var mine = S.compounds.filter(function (c) { return c.name.toLowerCase() === name.toLowerCase(); })[0];
                if (mine) { ids[k] = mine.id; return; }
                return api("POST", "/api/compounds", {
                    name: name, tag: fc.tag || "", volumeUl: fc.volumeUl, color: fc.color || nextColor()
                }).then(function (res) { ids[k] = res.compound.id; created++; applyState(res.state); });
            });
        });
        // 2. the steps, with this computer's compound ids (the server checks wells and timers again)
        return chain.then(function () {
            var steps = data.steps.map(function (st) {
                if (st && st.kind === "wait") return { kind: "wait", seconds: st.seconds };
                if (st && st.kind === "add" && ids[st.compound] != null) return { kind: "add", compoundId: ids[st.compound], wells: st.wells };
                return null;
            }).filter(Boolean);
            return api("PUT", "/api/plan", { plan: steps });
        }).then(function (res) {
            S.staged = {};
            applyState(res.state, true);
            toast("Imported " + res.steps + (res.steps === 1 ? " step" : " steps") +
                (created ? ", added " + created + (created === 1 ? " new substance" : " new substances") : "") + "." +
                (res.runReset ? " The run was reset." : ""));
        });
    }

    function renderBar() {
        document.body.dataset.tool = S.tool;
        [].forEach.call($("#toolSeg").children, function (b) { b.setAttribute("aria-pressed", String(b.dataset.tool === S.tool)); });
    }

    /* ---------- tabs ---------- */
    function setView(v) {
        S.view = v;
        document.body.dataset.view = v;
        [].forEach.call(document.querySelectorAll(".tab"), function (t) { t.setAttribute("aria-selected", String(t.dataset.tab === v)); });
        $("#viewPlate").hidden = v === "records";
        $("#viewRecords").hidden = v !== "records";
        if (v === "run" && S.editingId != null) stopEditing();
        closeMenu();
        $("#wellInfo").hidden = true;
        if (v === "records") loadPayload();
        renderAll();
        if (v !== "records") { fitGrid(); updateCells(); }
    }
    [].forEach.call(document.querySelectorAll(".tab"), function (t) { t.onclick = function () { setView(t.dataset.tab); }; });

    /* ---------- run ---------- */
    // Acts as the camera: "a tip with this compound just entered these wells".
    function testDrop(ids, cid) {
        if (cid == null) { toast("Select a substance first."); return; }
        if (S.run.status === "draft") { toast("Press Start run first."); return; }
        if (S.run.alarm) { toast("Acknowledge the warning first."); return; }
        api("POST", "/api/testDrop", { wells: ids, compoundId: cid }).then(function (res) {
            applyState(res.state);
            if (res.accepted === false) toast(res.reason === "runNotStarted" ? "Press Start run first." : "Select a substance first.");
            else if (res.complete && !res.alarm) toast("Run complete. Every well matches the plan.");
        }).catch(fail);
    }
    $("#recordBox").onchange = function (e) {
        var on = e.target.checked;
        api("POST", "/api/record", { record: on }).then(function (res) {
            applyState(res.state);
            toast(on ? "Each run is recorded from Start run (needs the camera running)." : "Runs are no longer recorded.");
        }).catch(fail);
    };
    $("#modeSeg").onclick = function (e) {
        var b = e.target.closest("button");
        if (!b || b.dataset.mode === S.mode) return;
        api("POST", "/api/runMode", { mode: b.dataset.mode }).then(function (res) {
            applyState(res.state);
            toast(S.mode === "free" ? "Free run: tap the substance in the tip, its wells light up." : "Guided run: one step at a time.");
        }).catch(fail);
    };
    $("#startBtn").onclick = function () {
        api("POST", "/api/run/start", {}).then(function (res) {
            applyState(res.state);
            toast("Run started. Tap the substance in the tip on the left.");
        }).catch(fail);
    };
    $("#resetBtn").onclick = function () {
        if (!window.confirm("Reset the run? Dispensed marks and the detection log are cleared.")) return;
        api("POST", "/api/run/reset").then(function (res) { applyState(res.state); toast("Run reset."); }).catch(fail);
    };
    $("#nextWellBtn").onclick = function () {
        var well = freeRun() ? readyFor(S.activeId)[0] : nextWellOfStep(currentStep());
        if (!well) return;
        var body = freeRun() ? { well: well, compoundId: S.activeId } : { well: well };
        api("POST", "/api/run/nextWell", body).then(function (res) {
            applyState(res.state);
            var cur = currentStep(), next = freeRun() ? readyFor(S.activeId)[0] : nextWellOfStep(cur);
            toast(well + " dispensed (manual)." + (next ? " Next well: " + next + "."
                : cur == null ? " Every well is done." : " Timer running."));
        }).catch(function (err) {
            if (err && err.message === "Not Found") toast("The server is running an older version. Stop python app.py (Ctrl+C) and start it again.");
            else fail(err);
        });
    };
    $("#cornerSelect").onchange = function (e) {
        api("POST", "/api/a1Corner", { a1Corner: e.target.value }).then(function (res) {
            applyState(res.state);
            toast("Camera told: A1 is " + e.target.options[e.target.selectedIndex].text.toLowerCase() + ".");
        }).catch(fail);
    };
    // A1 corner: click a corner of the small plate (or "let the camera guess"); same request as the select
    [].forEach.call(document.querySelectorAll("[data-corner]"), function (b) {
        b.onclick = function () {
            var sel = $("#cornerSelect");
            sel.value = b.dataset.corner;
            sel.dispatchEvent(new Event("change"));
        };
    });
    // Flip the camera picture: the camera code picks it up within a few seconds, even while running
    function flipCamera(key) {
        var body = {};
        body[key] = !(S.cameraFlip && S.cameraFlip[key]);
        api("POST", "/api/cameraFlip", body).then(function (res) {
            applyState(res.state);
            toast((key === "flipH" ? "Left\u2013right" : "Up\u2013down") + " flip " + (body[key] ? "on." : "off.")
                + (S.live && S.live.running ? " The camera picks it up in a few seconds." : ""));
        }).catch(fail);
    }
    $("#flipHBtn").onclick = function () { flipCamera("flipH"); };
    $("#flipVBtn").onclick = function () { flipCamera("flipV"); };
    $("#ackBtn").onclick = function () {
        api("POST", "/api/alarm/ack").then(function (res) { applyState(res.state); }).catch(fail);
    };
    function testTarget() {  // the well the current step needs next, and its compound
        var cur = currentStep();
        if (cur == null) { toast("Every step is done."); return null; }
        var w = nextWellOfStep(cur);
        if (!w) { toast("The wells left are on their timers. Wait for one to finish."); return null; }
        return { id: w, cid: S.steps[cur].compoundId };
    }
    $("#testOkBtn").onclick = function () {
        if (S.run.alarm) { toast("Acknowledge the warning first."); return; }
        var n = testTarget();
        if (n) testDrop([n.id], n.cid);
    };
    $("#testBadBtn").onclick = function () {
        if (S.run.alarm) { toast("Acknowledge the warning first."); return; }
        var n = testTarget();
        if (!n) return;
        var others = S.compounds.filter(function (c) { return c.id !== n.cid; });
        if (others.length) testDrop([n.id], others[Math.floor(Math.random() * others.length)].id);
        else toast("Add a second substance to test a wrong drop.");
    };

    /* ---------- sounds: synthesized with Web Audio (no files), all through one compressor ----------
       go (tip over the right well): a bright rising bell pair, "dispense here"
       warn (staying on a wrong well): a low two-pulse buzzer, repeated while the tip stays
       dispensed (a drop was counted): a quick rising arpeggio + a soft low pop
       complete (every well done): a short fanfare; alarm (a mistake was logged): an urgent siren
       chime (a well's timer ended): two soft bells */
    var actx, master;
    function audio() {
        if (!$("#soundBox").checked) return null;
        try {
            if (!actx) {
                actx = new (window.AudioContext || window.webkitAudioContext)();
                master = actx.createDynamicsCompressor();  // glues the layers together and keeps the peaks safe
                master.connect(actx.destination);
            }
            if (actx.state === "suspended") actx.resume();
            return actx;
        } catch (e) { return null; }
    }
    // One note. f: Hz, or [from, to] for a glide. at: start (s from now). o: type, gain, attack, lowpass, detune.
    function note(f, at, len, o) {
        var a = audio();
        if (!a) return;
        o = o || {};
        var t0 = a.currentTime + (at || 0), osc = a.createOscillator(), g = a.createGain(), out = g;
        osc.type = o.type || "sine";
        if (Array.isArray(f)) {
            osc.frequency.setValueAtTime(f[0], t0);
            osc.frequency.exponentialRampToValueAtTime(f[1], t0 + len);
        } else osc.frequency.setValueAtTime(f, t0);
        if (o.detune) osc.detune.value = o.detune;
        g.gain.setValueAtTime(0.0001, t0);
        g.gain.exponentialRampToValueAtTime(o.gain || 0.15, t0 + (o.attack || 0.005));
        g.gain.exponentialRampToValueAtTime(0.0001, t0 + len);
        osc.connect(g);
        if (o.lowpass) {
            var flt = a.createBiquadFilter();
            flt.type = "lowpass";
            flt.frequency.value = o.lowpass;
            g.connect(flt);
            out = flt;
        }
        out.connect(master);
        osc.start(t0);
        osc.stop(t0 + len + 0.05);
    }
    function bell(f, at, len, gain) {  // glassy: the note + two quiet inharmonic partials
        note(f, at, len, { gain: gain });
        note(f * 2.76, at, len * 0.5, { gain: gain * 0.22 });
        note(f * 5.4, at, len * 0.25, { gain: gain * 0.07 });
    }
    function goSound() { bell(1046.5, 0, 0.35, 0.16); bell(1568, 0.09, 0.5, 0.16); }  // C6 -> G6
    function warnSound() {  // dissonant low buzz, falling: "no"
        [0, 0.2].forEach(function (at) {
            note([233, 196], at, 0.17, { type: "sawtooth", gain: 0.13, lowpass: 1500 });
            note([247, 207], at, 0.17, { type: "sawtooth", gain: 0.1, lowpass: 1500 });
        });
    }
    function dispensedSound() {  // C6 E6 G6 C7 arpeggio + a soft pop underneath
        note([190, 60], 0, 0.14, { gain: 0.22 });
        [1046.5, 1318.5, 1568, 2093].forEach(function (f, k) { bell(f, 0.03 + k * 0.06, 0.45, 0.11); });
    }
    function completeSound() {  // fanfare: C5 E5 G5 C6, then the chord rings
        [523.25, 659.25, 784, 1046.5].forEach(function (f, k) { bell(f, k * 0.12, 0.5, 0.13); });
        [523.25, 784, 1318.5].forEach(function (f) { note(f, 0.5, 1.6, { type: "triangle", gain: 0.06, attack: 0.04 }); });
    }
    function chime() { bell(880, 0, 0.6, 0.1); bell(1318.5, 0.22, 0.8, 0.1); }  // a timer finished: A5 -> E6
    function beep() {  // a logged mistake: urgent two-tone siren, 3 times
        [0, 0.18, 0.36].forEach(function (at, k) {
            note(k % 2 ? 660 : 880, at, 0.16, { type: "square", gain: 0.08, lowpass: 2600 });
        });
    }

    /* ---------- live tip (camera or Camera Simulator) ----------
       The server says which well the tip is over and what a drop there would be for the compound in the tip:
       ok = green ring + the go sound, a mistake = red ring; staying TIP_WARN_S on a wrong well = about to
       dispense: flashing + warning sound, before anything is dropped. */
    var TIP_WARN_S = 0.7;
    var TIP_TEXT = {
        ok: function (t) { return "Right well for " + t.compound; },
        added: function (t) { return t.compound + " added. Move on to the next well."; },
        wrong: function (t) { return "Wrong well: it needs " + t.planned + ", not " + t.compound; },
        unplanned: function () { return "Wrong well: nothing is planned here"; },
        duplicate: function (t) { return "Wrong well: " + t.compound + " is already in it"; },
        order: function (t) { return "Wrong order: it needs " + t.planned + " first"; },
        early: function () { return "Too early: the timer before this step is still running"; }
    };
    var tipEnteredAt = 0, tipWarnedAt = 0, lastDing = {};
    S.tip = { well: null };
    function tipBad(t) { return !!(t && t.well && t.kind && t.kind !== "ok" && t.kind !== "added"); }
    function tipState() {
        var t = S.tip;
        return !t.well ? "off" : !t.kind ? "idle" : tipBad(t) ? (S.tipDanger ? "danger" : "bad") : "ok";
    }
    function setTip(t, fromState) {
        t = t || { well: null };
        var prev = S.tip, now = performance.now();
        if (t.well !== prev.well) {
            tipEnteredAt = now;
            tipWarnedAt = 0;
            S.tipDanger = false;
        }
        var turnedRight = t.kind === "ok" && (t.well !== prev.well || prev.kind !== "ok");
        if (turnedRight && now - (lastDing[t.well] || 0) > 1500) { goSound(); lastDing[t.well] = now; }
        S.tip = t;
        if (!tipBad(t)) S.tipDanger = false;
        if (!fromState) { updateCells(); renderTip(); }  // a full state renders everything anyway
    }
    setInterval(function () {
        if (!tipBad(S.tip) || S.view !== "run") return;
        var danger = (performance.now() - tipEnteredAt) / 1000 >= TIP_WARN_S;
        if (danger !== !!S.tipDanger) { S.tipDanger = danger; updateCells(); renderTip(); }
        if (danger && performance.now() - tipWarnedAt > 1200) { warnSound(); tipWarnedAt = performance.now(); }
    }, 100);
    function renderTip() {
        var t = S.tip, bar = $("#tipBar");
        // The bar keeps its space in the Run tab (so the plate is the same size as in Plan); it just hides its content
        var idle = !t.well && !(S.camera && S.camera.online);
        bar.classList.toggle("idle", idle);
        if (idle) return;
        bar.dataset.state = tipState();
        $("#tipWell").textContent = t.well || "";
        var text = !t.well ? "Not over a well"
            : !t.kind ? (S.run.status === "draft" ? "Press Start run to check wells" : "Pick the substance in the tip on the left")
            : (TIP_TEXT[t.kind] || function () { return t.kind; })(t);
        $("#tipText").textContent = (S.tipDanger ? "STOP, about to dispense. " : "") + text;
    }
    var PILL = { draft: "Not started", active: "Running", complete: "Complete", alarm: "Check well", wait: "Timer running" };
    function stat(label, valueEl) { var d = el("div", "runStat"); d.appendChild(el("span", "runLabel", label)); d.appendChild(valueEl); return d; }
    function renderRun() {
        var status = S.run.status, cur = currentStep(), curSt = cur == null ? null : S.steps[cur];
        var p = S.run.progress || { done: 0, total: 0 }, total = 0, d = 0;
        S.steps.forEach(function (st, i) { if (st.kind === "add") st.wells.forEach(function (w) { total++; if (stepDone(i, w)) d++; }); });
        var free = freeRun(), ready = free ? readyFor(S.activeId) : [];
        var nextW = free ? ready[0] || null : nextWellOfStep(cur);
        var held = nextW ? null : free ? heldFor(S.activeId) : waitingTimer(cur);
        var pill = $("#runPill"), state = S.run.alarm ? "alarm" : held ? "wait" : status;
        pill.dataset.state = state;
        pill.textContent = PILL[state];
        $("#stepNow").textContent = !running() ? S.steps.length + (S.steps.length === 1 ? " step" : " steps")
            : cur == null ? "All steps done" : free ? "Free run: any substance" : "Step " + (cur + 1) + " of " + S.steps.length;
        $("#recPill").hidden = !(S.camera && S.camera.recording);
        $("#recPill").title = S.camera && S.camera.recording ? "Recording: " + S.camera.recording : "";
        [].forEach.call($("#modeSeg").children, function (b) { b.setAttribute("aria-pressed", String(b.dataset.mode === S.mode)); });

        var pt = $("#progText");
        pt.replaceChildren();
        var v = el("span", "runValue");
        v.appendChild(el("b", null, d + " / " + total));
        v.appendChild(el("span", "muted", "additions"));
        pt.appendChild(v);
        $("#progBar").style.width = total && running() ? (100 * d / total) + "%" : "0";

        // In tip: what the user says is in the tip, and a warning if the current step needs another compound
        var c = byId(S.activeId), tip = $("#inTip");
        tip.replaceChildren(el("span", "runLabel", "In tip"));
        // The box takes the compound's own color, so it matches its wells on the plate
        if (c) tip.style.setProperty("--tipColor", c.color); else tip.style.removeProperty("--tipColor");
        tip.classList.toggle("empty", !c);
        var tv = el("span", "bigValue");
        if (c) {
            tv.appendChild(el("span", "bigName", c.name));
            tv.appendChild(el("span", "muted", fmtVol(c.volumeUl)));
        } else tv.appendChild(el("span", "muted", "Tap a substance on the left"));
        tip.appendChild(tv);
        var warn = null;
        if (free) {  // free run: warn only when no well can take this substance now
            if (c && running() && cur != null && !ready.length) warn = held ? "Wells on a timer" : "No well needs this now";
        } else if (curSt && curSt.kind === "add" && curSt.compoundId !== S.activeId && byId(curSt.compoundId)) {
            warn = "Step " + (cur + 1) + " needs " + byId(curSt.compoundId).name;
        }
        if (warn) tip.appendChild(el("span", "tipWarn", warn));
        tip.classList.toggle("wrong", !!warn);

        // Next: the well, or the timer countdown
        var nx = $("#nextWell"), nv = el("span", "bigValue");
        nx.classList.toggle("timer", !!held);
        nx.classList.toggle("idle", !held && (!running() || cur == null));
        if (held) {  // every well left in this step is still on its timer
            nx.replaceChildren(el("span", "runLabel", "Timer"));
            nv.appendChild(el("span", "nextId timer", fmtTime(held.left)));
            var heldC = free ? byId(S.activeId) : byId(curSt.compoundId);
            nv.appendChild(el("span", "muted", "then " + held.well + ": " + (heldC || { name: "?" }).name));
        } else if (free && running() && cur != null) {  // free run: how many wells take what is in the tip
            nx.replaceChildren(el("span", "runLabel", ready.length === 1 ? "Well for the tip" : "Wells for the tip"));
            nv.appendChild(el("span", "nextId", ready.length ? String(ready.length) : "\u2013"));
            nv.appendChild(el("span", "muted", ready.length ? ready.slice(0, 4).join(", ") + (ready.length > 4 ? " \u2026" : "")
                : "pick another substance"));
        } else {
            nx.replaceChildren(el("span", "runLabel", "Next well"));
            if (!running()) nv.appendChild(el("span", "muted", "Press Start run"));
            else if (cur == null) nv.appendChild(el("span", "muted", "Done"));
            else nv.appendChild(el("span", "nextId", nextW || "\u2013"));
        }
        nx.appendChild(nv);

        var start = $("#startBtn");
        start.hidden = status === "active";
        start.textContent = status === "draft" ? "Start run" : "Start new run";
        start.disabled = !total;
        $("#resetBtn").hidden = status !== "active";
        $("#nextWellBtn").hidden = status !== "active" || !nextW;
        $("#testOkBtn").disabled = $("#testBadBtn").disabled = !running() || !total;

        var sum = $("#setupSummary"), corner = $("#cornerSelect");
        $("#a1Badge").hidden = S.a1Corner !== "auto";  // the button keeps its name; a badge says the corner is not set
        sum.title = S.a1Corner === "auto" ? "Well A1 corner not set" : "A1: " + corner.options[corner.selectedIndex].text.toLowerCase();
        [].forEach.call(document.querySelectorAll("[data-corner]"), function (b) { b.setAttribute("aria-pressed", String(b.dataset.corner === S.a1Corner)); });
        var fl = S.cameraFlip || {};
        $("#flipHBtn").setAttribute("aria-pressed", String(!!fl.flipH));
        $("#flipVBtn").setAttribute("aria-pressed", String(!!fl.flipV));
    }
    // Well timers count down on the page (rings updated in place every 0.5 s); a well whose timer ends chimes.
    var timersSeen = {}, ringWells = "";
    setInterval(function () {
        if (!running()) { timersSeen = {}; return; }
        var now = [], ready = [], all = S.run.wellTimers || {};
        Object.keys(all).forEach(function (w) {
            all[w].forEach(function (t) {
                var x = liveTimer(t), key = S.serverId + ":" + S.run.startedAt + ":" + w + ":" + t.step;
                if (x.state === "running") { now.push(w); timersSeen[key] = "running"; }
                if (x.state === "done" && timersSeen[key] === "running") { timersSeen[key] = "done"; ready.push(w); }
            });
        });
        if (ready.length) {
            chime();
            toast(ready.length === 1 ? ready[0] + " timer done: it is ready for its next substance." : ready.length + " wells ready: " + ready.sort(colMajor).join(", ") + ".");
        }
        var key = now.join(",");
        if (key !== ringWells || ready.length) { ringWells = key; updateCells(); }  // a timer started or ended
        else cellEls.forEach(function (w) {  // only the time moved: update the rings in place
            var tm = runningTimerOf(w.dataset.id), ring = w.querySelector(".timerRing"), lbl = w.querySelector(".timerLeft");
            if (tm && ring && lbl) { ring.style.setProperty("--p", tm.left / tm.seconds); lbl.textContent = fmtTime(tm.left); }
        });
        if (now.length || ready.length) { renderRun(); renderSteps(); }
    }, 500);
    function renderBanner() {
        var a = S.run.alarm, b = $("#banner");
        b.hidden = !a;
        if (!a) return;
        $("#bannerTitle").textContent = TITLES[a.kind] || "Warning";
        var others = a.wells.filter(function (w) { return w !== (a.well || a.wells[0]); });
        var extra = others.length ? " Also flagged: " + others.join(", ") + "." : "";
        $("#bannerMsg").textContent = a.message + extra;
    }
    function renderLink() {
        $("#serverDot").dataset.state = S.connected ? "ok" : "warn";
        $("#serverText").textContent = S.connected ? "Server: live" : "Server: reconnecting…";
        var cam = S.camera || {}, dot = $("#aiDot"), txt = $("#aiText");
        if (cam.online) { dot.dataset.state = "ok"; txt.textContent = "Camera: online"; }
        else if (cam.lastSeen) { dot.dataset.state = "warn"; txt.textContent = "Camera: offline (last seen " + clock(cam.lastSeen) + ")"; }
        else { dot.dataset.state = ""; txt.textContent = "Camera: not connected"; }
    }

    /* ---------- records ---------- */
    function loadPayload() {
        api("GET", "/api/plan").then(function (p) {
            payloadText = JSON.stringify(p, null, 2);
            if (S.view === "records") $("#payload").textContent = payloadText;
        }).catch(function () {});
    }
    function renderRecords() {
        if (S.view !== "records") return;
        var body = $("#logBody");
        body.replaceChildren();
        S.events.forEach(function (ev) {
            var tr = el("tr");
            tr.appendChild(el("td", null, clock(ev.time)));
            tr.appendChild(el("td", null, ev.well));
            tr.appendChild(el("td", null, ev.plannedName));
            tr.appendChild(el("td", null, ev.detectedName));
            var td = el("td");
            td.appendChild(el("span", "chip " + (ev.kind === "ok" ? "ok" : "bad"), LABELS[ev.kind] || ev.kind));
            tr.appendChild(td);
            tr.appendChild(el("td", null, SOURCES[ev.source] || ev.source));
            body.appendChild(tr);
        });
        $("#logEmpty").hidden = S.events.length > 0;
        $("#payload").textContent = payloadText;
    }
    $("#copyBtn").onclick = function () {
        try {
            navigator.clipboard.writeText(payloadText).then(
                function () { toast("Plan JSON copied."); },
                function () { toast("Copy blocked. Select the text and copy it manually."); });
        } catch (e) { toast("Copy blocked. Select the text and copy it manually."); }
    };

    /* ---------- live camera (Start camera: the server runs scripts/live_demo.py) ---------- */
    var liveBusy = false, liveSeenExit = null;
    try { $("#cameraInput").value = localStorage.getItem("pipetteCoach.camera") || "0"; } catch (e) { /* storage blocked */ }
    function renderLive() {
        var L = S.live || {}, btn = $("#liveBtn");
        btn.classList.toggle("on", !!L.running);
        btn.disabled = liveBusy;
        $("#liveText").textContent = L.running
            ? (S.camera && S.camera.online ? "Camera live · Stop" : "Camera starting… · Stop")
            : "Start camera";
        var key = L.running ? null : L.exitCode;  // tell once when it stopped by itself with an error
        if (key != null && key !== 0 && key !== -15 && liveSeenExit !== key + ":" + (L.log || []).join("|")) {
            liveSeenExit = key + ":" + (L.log || []).join("|");
            toast("The camera stopped: " + ((L.log || []).slice(-1)[0] || "exit code " + key) + " (full log: data/liveCamera.log)");
        }
    }
    // Find cameras: the server opens each camera for a moment and sends a small snapshot, so you can tell them apart
    var cameras = [];
    function renderCameras() {
        var list = $("#camList"), chosen = $("#cameraInput").value.trim();
        list.innerHTML = "";
        cameras.forEach(function (c) {
            var b = document.createElement("button"), img = document.createElement("img"), t = document.createElement("b"), d = document.createElement("span");
            b.type = "button";
            b.className = "camOpt";
            b.setAttribute("aria-pressed", String(chosen === String(c.index)));
            img.src = c.thumb;
            img.alt = "";
            t.textContent = "Camera " + c.index + (c.name ? " \u00B7 " + c.name : "");
            d.textContent = c.width + " \u00D7 " + c.height;
            b.append(img, t, d);
            b.onclick = function () { setCamera(String(c.index)); };
            list.append(b);
        });
    }
    function setCamera(v) {
        $("#cameraInput").value = v;
        try { localStorage.setItem("pipetteCoach.camera", v); } catch (e) { /* storage blocked */ }
        renderCameras();
    }
    $("#cameraInput").oninput = function () { setCamera($("#cameraInput").value.trim()); };
    $("#findCamBtn").onclick = function () {
        var btn = $("#findCamBtn");
        btn.disabled = true;
        btn.textContent = "Looking\u2026";
        api("GET", "/api/live/cameras").then(function (res) {
            cameras = res.cameras || [];
            renderCameras();
            toast(cameras.length ? cameras.length + (cameras.length === 1 ? " camera found." : " cameras found. Click the one that sees the plate.")
                : "No camera found. Is one plugged in, and is camera access allowed for Terminal?");
        }).catch(fail).then(function () { btn.disabled = false; btn.textContent = "Find cameras"; });
    };

    $("#liveBtn").onclick = function () {
        var running = S.live && S.live.running, cam = $("#cameraInput").value.trim() || "0";
        try { localStorage.setItem("pipetteCoach.camera", cam); } catch (e) { /* storage blocked */ }
        liveBusy = true;
        renderLive();
        var call = running ? api("POST", "/api/live/stop")
            : api("POST", "/api/live/start", { camera: cam, show: $("#liveWindowBox").checked });
        call.then(function (res) {
            liveBusy = false;
            applyState(res.state);
            toast(running ? "Camera stopped." : "Starting camera " + cam + "… the models take a few seconds to load.");
        }).catch(function (err) { liveBusy = false; renderLive(); fail(err); });
    };

    function renderAll() {
        renderList();
        renderEditorState();
        updateCells();
        renderBar();
        renderStage();
        renderSteps();
        renderRun();
        renderBanner();
        renderTip();
        renderLink();
        renderLive();
        renderRecords();
    }

    /* ---------- start ---------- */
    setColor(PALETTE[0]);
    rebuildGrid();
    if (/^#(run|records)$/.test(location.hash)) setView(location.hash.slice(1));  // e.g. open /#run on the lab tablet
    renderAll();
    reloadState();
    connect();
    // If the live connection is down, keep the page fresh by asking every 4 s.
    setInterval(function () { if (!S.connected) reloadState(); }, 4000);
    setTimeout(function () { fitGrid(); updateCells(); }, 60);
})();
