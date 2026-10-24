/* Camera Simulator (/camera): plays the camera for testing the Plate Tracker without a plate.
   Hover = tip over a well (nothing is sent, like a real hover). Click = hold the tip there. Dispense = POST
   /api/sim/drop, judged by the server exactly like a camera drop. A heartbeat keeps "Camera: online" on the
   Plate Tracker while this tab is open. */
(function () {
    "use strict";
    var ROWS = 8, COLS = 12;
    var LABELS = { ok: "Match", wrong: "Wrong compound", unplanned: "Unplanned", duplicate: "Duplicate", order: "Out of order", early: "Too early" };
    var REASONS = { runNotStarted: "Not counted: press Start run on the Plate Tracker.", noActiveCompound: "Not counted: pick the compound in the tip on the Plate Tracker." };
    var $ = function (s) { return document.querySelector(s); };
    var S = { state: null, hover: null, sel: null, liquid: {}, startedAt: null, cells: {}, tip: { well: null }, sentTip: undefined };
    var VERDICT = {
        ok: function (t) { return "right well for " + t.compound; },
        added: function (t) { return t.compound + " added"; },
        wrong: function (t) { return "wrong: needs " + t.planned; },
        unplanned: function () { return "wrong: nothing planned"; },
        duplicate: function (t) { return "wrong: " + t.compound + " already in"; },
        order: function (t) { return "wrong order: needs " + t.planned + " first"; },
        early: function () { return "too early: timer running"; }
    };

    function wid(r, c) { return String.fromCharCode(65 + r) + (c + 1); }
    function el(tag, cls, text) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text != null) e.textContent = text;
        return e;
    }
    function api(method, url, body) {
        var opts = { method: method, headers: {} };
        if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
        return fetch(url, opts).then(function (res) {
            return res.json().catch(function () { return {}; }).then(function (data) {
                if (!res.ok) throw new Error(res.status === 404 ? "The server is running an older version. Restart python app.py."
                    : data.error || "Server error " + res.status);
                return data;
            });
        }, function () { throw new Error("Cannot reach the Plate Tracker server."); });
    }

    /* ---------- the plate ---------- */
    var plate = $("#plate");
    plate.appendChild(el("span"));
    for (var c = 0; c < COLS; c++) plate.appendChild(el("span", "ax", String(c + 1)));
    for (var r = 0; r < ROWS; r++) {
        plate.appendChild(el("span", "ax", String.fromCharCode(65 + r)));
        for (c = 0; c < COLS; c++) {
            var w = el("div", "simWell");
            w.dataset.id = wid(r, c);
            w.title = w.dataset.id;
            plate.appendChild(w);
            S.cells[w.dataset.id] = w;
        }
    }
    plate.addEventListener("mouseover", function (e) {
        var w = e.target.closest(".simWell");
        S.hover = w ? w.dataset.id : null;
        render();
    });
    plate.addEventListener("mouseleave", function () { S.hover = null; render(); });
    plate.addEventListener("click", function (e) {
        var w = e.target.closest(".simWell");
        if (!w) return;
        S.sel = S.sel === w.dataset.id ? null : w.dataset.id;
        render();
    });

    function lift() { S.sel = null; render(); }
    $("#liftBtn").onclick = lift;
    $("#emptyBtn").onclick = function () { S.liquid = {}; render(); };
    $("#showPlan").onchange = render;
    $("#dispenseBtn").onclick = dispense;
    document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") lift();
        else if ((e.key === "d" || e.key === "D" || e.key === "Enter") && S.sel) { e.preventDefault(); dispense(); }
    });

    function dispense() {
        var well = S.sel;
        if (!well) return;
        S.liquid[well] = (S.liquid[well] || 0) + 1;  // the liquid is in the well whatever the tracker says
        render();
        api("POST", "/api/sim/drop", { wells: [well] }).then(function (res) {
            if (res.state) applyState(res.state);
            if (res.accepted === false) addLog(well, "info", REASONS[res.reason] || res.reason);
            else {
                var r = res.results[0];
                addLog(well, r.kind === "ok" ? "ok" : "bad", LABELS[r.kind] || r.kind, r.kind === "unplanned" ? "" : "planned: " + r.planned);
            }
        }).catch(function (err) { addLog(well, "bad", "Not sent", err.message); });
    }

    var logEntries = 0;
    function addLog(well, cls, label, note) {
        var log = $("#log");
        if (!logEntries++) log.replaceChildren();
        var li = el("li");
        li.appendChild(el("span", "t", new Date().toLocaleTimeString()));
        li.appendChild(el("b", null, well));
        li.appendChild(el("span", "chip " + cls, label));
        if (note) li.appendChild(el("span", "hint", note));
        log.prepend(li);
    }

    /* ---------- what the Plate Tracker knows ---------- */
    function compound(id) { return (S.state.compounds || []).filter(function (c) { return c.id === id; })[0]; }
    function nextFor(well) {  // the compound this well still needs next (to draw the plan)
        var st = S.state, steps = st.plate.steps, done = st.run.done || {};
        for (var i = 0; i < steps.length; i++) {
            if (steps[i].kind !== "add" || steps[i].wells.indexOf(well) < 0) continue;
            if (!(st.run.status !== "draft" && done[String(i)] && done[String(i)][well])) return compound(steps[i].compoundId);
        }
        return null;
    }
    function applyState(st) {
        if (!st || !st.plate || !Array.isArray(st.plate.steps)) return;
        if (st.tip) S.tip = st.tip;
        if (S.startedAt !== st.run.startedAt) { S.liquid = {}; S.startedAt = st.run.startedAt; }  // a new run: clean plate
        S.state = st;
        render();
    }

    function render() {
        var st = S.state, showPlan = $("#showPlan").checked;
        Object.keys(S.cells).forEach(function (id) {
            var w = S.cells[id], need = st && showPlan ? nextFor(id) : null, n = S.liquid[id] || 0;
            w.classList.toggle("planned", !!need);
            w.style.borderColor = need ? need.color : "";
            w.classList.toggle("liquid", n > 0);
            w.classList.toggle("sel", S.sel === id);
            w.classList.toggle("hover", S.hover === id);
            w.textContent = n > 1 ? n + "×" : need ? need.tag : "";
            w.title = id + (need ? " · needs " + need.name : "") + (n ? " · dispensed " + n + "×" : "");
        });
        var tipWell = S.hover || S.sel || null;  // the mouse is the tip; off the plate it rests on the selected well
        if (tipWell !== S.sentTip) {
            S.sentTip = tipWell;
            api("POST", "/api/sim/tip", { well: tipWell }).catch(function () {});
        }
        var hv = $("#hoverWell"), sv = $("#selWell"), t = S.tip.well === tipWell ? S.tip : { well: tipWell };
        hv.textContent = tipWell ? tipWell + (S.hover ? "" : " (held)") : "off the plate";
        hv.classList.toggle("off", !tipWell);
        var vd = $("#verdict"), bad = !!(t.kind && t.kind !== "ok" && t.kind !== "added");
        vd.textContent = !tipWell ? "" : !t.kind ? "not checked (run not started or no compound in the tip)"
            : (VERDICT[t.kind] || function () { return t.kind; })(t);
        vd.dataset.state = !tipWell || !t.kind ? "" : bad ? "bad" : "ok";
        sv.textContent = S.sel || "click a well";
        sv.classList.toggle("off", !S.sel);
        var btn = $("#dispenseBtn");
        btn.disabled = !S.sel;
        btn.textContent = S.sel ? "Dispense into " + S.sel : "Dispense";
        if (!st) return;
        var RUN = { draft: "Not started", active: "Running", complete: "Complete" };
        $("#runState").textContent = RUN[st.run.status] + (st.run.alarm ? " · alarm" : "");
        var tip = $("#inTip"), c = compound(st.run.activeCompoundId);
        tip.replaceChildren();
        if (c) {
            var tag = el("span", "tipTag", c.tag);
            tag.style.background = c.color;
            tag.style.color = "#fff";
            tip.appendChild(tag);
            tip.appendChild(document.createTextNode(" " + c.name));
        } else tip.textContent = "none picked";
    }

    /* ---------- live link ---------- */
    function setLink(ok) { $("#link").textContent = ok ? "connected, sending as the camera" : "not reachable"; }
    api("GET", "/api/state").then(function (st) { setLink(true); applyState(st); }).catch(function () { setLink(false); });
    var es = new EventSource("/api/stream");
    es.addEventListener("state", function (e) { setLink(true); applyState(JSON.parse(e.data)); });
    es.addEventListener("tip", function (e) { S.tip = JSON.parse(e.data); render(); });
    es.onerror = function () { setLink(false); };
    window.addEventListener("pagehide", function () {  // tab closed: the tip leaves the plate
        if (navigator.sendBeacon) navigator.sendBeacon("/api/sim/tip", JSON.stringify({ well: null }));
    });
    function beat() { if (!document.hidden) api("POST", "/api/sim/heartbeat", {}).catch(function () {}); }
    beat();
    setInterval(beat, 2000);
    render();
})();
