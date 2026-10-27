/* Braindance Lab — BD editor
 * Plays "tapes" exported by notebook 07 (or scripts/make_demo_data.py).
 * Schema: docs/tape_schema.md. No build step, no dependencies.
 */
(() => {
  "use strict";

  const STAGE_NAMES = { W: "Awake", N1: "Drifting off", N2: "Light sleep", N3: "Deep sleep", REM: "Dreaming" };
  const STAGE_ROWS = ["W", "REM", "N1", "N2", "N3"];             // hypnogram convention, top to bottom
  const QUAD = {
    HVHA: { word: "Elated", plain: "pleasant and intense" },
    HVLA: { word: "Serene", plain: "pleasant and calm" },
    LVHA: { word: "Tense", plain: "unpleasant and intense" },
    LVLA: { word: "Gloomy", plain: "unpleasant and calm" },
  };
  const BAND_NAMES = { delta: "δ delta", theta: "θ theta", alpha: "α alpha", sigma: "σ sigma", beta: "β beta", gamma: "γ gamma" };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const stageColor = (s) => css(`--st-${s}`);
  const bandColor = (b) => css(`--${b}`);

  const state = {
    manifest: null, metrics: null, cache: new Map(),
    tape: null, pos: 0, playing: false, layer: "raw", smoothed: true,
    timer: null, scroll: 0, lastFrame: 0,
  };

  // ------------------------------------------------------------------ loading
  async function getJSON(path) {
    const res = await fetch(path, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${path}: ${res.status}`);
    return res.json();
  }

  async function init() {
    try {
      state.manifest = await getJSON("data/manifest.json");
    } catch (e) {
      $("readout-state").textContent = "No tapes";
      $("readout-sub").textContent = "Couldn't load data/manifest.json. Run notebook 07 (or scripts/make_demo_data.py) and copy its output into app/data/.";
      return;
    }
    try { state.metrics = await getJSON(`data/${state.manifest.metrics || "metrics.json"}`); } catch { state.metrics = null; }
    $("demo-banner").hidden = !state.manifest.demo;
    buildChips();
    bindControls();
    const first = state.manifest.tapes.find((t) => t.kind === "sleep") || state.manifest.tapes[0];
    if (first) await selectTape(first.id);
    requestAnimationFrame(frame);
  }

  function buildChips() {
    const groups = { sleep: $("sleep-tapes"), emotion: $("emotion-tapes") };
    const listeners = new Map();   // songs grouped per listener: "M01  1 2 3 4"
    for (const t of state.manifest.tapes) {
      const b = document.createElement("button");
      b.className = "chip";
      b.dataset.id = t.id;
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => selectTape(t.id));
      if (t.kind === "sleep") {
        b.textContent = t.title.replace(" · ", ", ");
        groups.sleep.appendChild(b);
      } else if (t.kind === "emotion") {
        b.classList.add("small");
        b.textContent = t.session;
        b.setAttribute("aria-label", `${t.subject}, song ${t.session}`);
        if (!listeners.has(t.subject)) {
          const row = document.createElement("span");
          row.className = "listener";
          row.innerHTML = `<span class="listener-name">${t.subject}</span>`;
          listeners.set(t.subject, row);
          groups.emotion.appendChild(row);
        }
        listeners.get(t.subject).appendChild(b);
      }
    }
    for (const [kind, el] of Object.entries(groups)) {
      if (!el.children.length) el.closest(".tape-group").hidden = true;
    }
  }

  async function selectTape(id) {
    const entry = state.manifest.tapes.find((t) => t.id === id);
    if (!state.cache.has(id)) state.cache.set(id, await getJSON(`data/${entry.file}`));
    stop();
    state.tape = state.cache.get(id);
    state.pos = 0;
    state.scroll = 0;
    document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.id === id)));
    if (state.tape.kind === "emotion") enrichEmotion(state.tape);
    // decorate emotion chips with the rated quadrant once loaded
    const chip = document.querySelector(`.chip[data-id="${id}"]`);
    if (state.tape.kind === "emotion" && chip && !chip.querySelector(".q")) {
      const dot = document.createElement("span");
      dot.className = "q";
      dot.style.background = quadColor(state.tape.rating.quadrant);
      chip.prepend(dot);
    }
    const sleep = state.tape.kind === "sleep";
    $("hypnogram").hidden = !sleep;
    $("circumplex-wrap").hidden = sleep;
    $("smooth-toggle-wrap").hidden = !sleep;
    $("timeline-title").textContent = sleep ? "The night, scored twice" : "Where the song lands";
    $("timeline-hint").textContent = sleep
      ? "Grey steps are the expert's scoring; colour is the model. Click to jump; arrow keys step, space plays."
      : "Dot: the model's reading of this 4 s window. Trail: its smoothed path so far. Ring: its verdict for the whole song. Star: what the listener reported.";
    renderAll();
  }

  function enrichEmotion(t) {
    if (t._ready) return;
    t._ready = true;
    t.windows.n = t.windows.t.length;
  }

  // ----------------------------------------------------------------- controls
  function bindControls() {
    $("play").addEventListener("click", () => (state.playing ? stop() : play()));
    document.querySelectorAll(".layers button").forEach((b) =>
      b.addEventListener("click", () => {
        state.layer = b.dataset.layer;
        document.querySelectorAll(".layers button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
        renderScope();
      }));
    $("smooth-toggle").addEventListener("change", (e) => { state.smoothed = e.target.checked; renderAll(); });
    $("hypnogram").addEventListener("click", (e) => {
      const rect = e.currentTarget.getBoundingClientRect();
      seek(Math.floor(((e.clientX - rect.left) / rect.width) * length()));
    });
    document.addEventListener("keydown", (e) => {
      if (!state.tape) return;
      const tag = e.target.tagName;
      if (e.key === " " && tag !== "BUTTON" && tag !== "INPUT") { e.preventDefault(); state.playing ? stop() : play(); }
      if (e.key === "ArrowRight") { e.preventDefault(); seek(state.pos + (e.shiftKey ? 20 : 1)); }
      if (e.key === "ArrowLeft") { e.preventDefault(); seek(state.pos - (e.shiftKey ? 20 : 1)); }
    });
    let resizeT;
    window.addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(renderAll, 120); });
    window.matchMedia("(prefers-color-scheme: light)").addEventListener?.("change", renderAll);
    // canvases read theme tokens at draw time, so redraw when an explicit theme toggle flips
    new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  const length = () => (state.tape.kind === "sleep" ? state.tape.epochs.t.length : state.tape.windows.t.length);

  function play() {
    if (state.pos >= length() - 1) state.pos = 0;
    state.playing = true;
    $("play").textContent = "Pause";
    $("play").setAttribute("aria-label", "Pause");
    const step = state.tape.kind === "sleep" ? 350 : 650;
    state.timer = setInterval(() => {
      if (state.pos >= length() - 1) return stop();
      seek(state.pos + 1, true);
    }, step);
  }

  function stop() {
    state.playing = false;
    clearInterval(state.timer);
    $("play").textContent = "Play";
    $("play").setAttribute("aria-label", "Play");
  }

  function seek(p, fromPlay = false) {
    if (!state.tape) return;
    const next = Math.max(0, Math.min(length() - 1, p));
    if (next !== state.pos) state.scroll = 0;
    state.pos = next;
    if (!fromPlay && state.playing) { /* keep playing from the new spot */ }
    renderReadout();
    renderTimeline();
    renderScope();
  }

  // ---------------------------------------------------------------- rendering
  function renderAll() {
    if (!state.tape) return;
    renderReadout();
    renderTimeline();
    renderScope();
    renderRadar();
    renderMetrics();
  }

  function predAt(i) {
    const e = state.tape.epochs;
    return (state.smoothed ? e.pred : e.pred_raw)[i];
  }

  function renderReadout() {
    const t = state.tape;
    const big = $("readout-state");
    const sub = $("readout-sub");
    let color;
    if (t.kind === "sleep") {
      const e = t.epochs;
      const truth = t.stages[e.true[state.pos]];
      const p = predAt(state.pos);
      if (e.artifact[state.pos]) {
        big.textContent = "Static";
        sub.textContent = `This window is flagged as an artifact — movement or a loose electrode. The expert scored ${truth}.`;
        color = css("--muted");
      } else if (p < 0) {
        big.textContent = truth;
        sub.textContent = `${STAGE_NAMES[truth]} · no model reading for this window.`;
        color = stageColor(truth);
      } else {
        const ps = t.stages[p];
        big.textContent = ps;
        const agree = ps === truth ? "the expert agrees" : `the expert scored ${truth}`;
        sub.textContent = `${STAGE_NAMES[ps]} · ${Math.round(e.conf[state.pos] * 100)}% confident · ${agree}.`;
        color = stageColor(ps);
      }
    } else {
      const w = t.windows;
      const pv = w.p_valence[state.pos], pa = w.p_arousal[state.pos];
      const q = `${pv >= 0.5 ? "H" : "L"}V${pa >= 0.5 ? "H" : "L"}A`;
      if (pv == null || pa == null) {
        big.textContent = "—";
        sub.textContent = "No model reading for this window.";
        color = css("--muted");
      } else {
        big.textContent = QUAD[q].word;
        const rq = t.rating.quadrant;
        const told = rq ? `they rated the song ${QUAD[rq].plain}` : "their rating was right on the midpoint";
        sub.textContent = `Reads as ${QUAD[q].plain} (${pct(pv)} pleasant, ${pct(pa)} intense) · ${told}.`;
        color = quadColor(q);
      }
    }
    document.documentElement.style.setProperty("--state-color", color);
    const secs = t.kind === "sleep" ? t.epochs.t[state.pos] * 60 : t.windows.t[state.pos];
    $("clock").textContent = t.kind === "sleep"
      ? `${fmtClock(secs)} into the recording · epoch ${state.pos + 1} of ${length()} · 30 s each`
      : `${secs.toFixed(0)} s into the song · window ${state.pos + 1} of ${length()} · 4 s each`;
  }

  const pct = (p) => `${Math.round(p * 100)}%`;
  const fmtClock = (s) => `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
  function quadColor(q) {
    return { HVHA: css("--theta"), HVLA: css("--alpha"), LVHA: css("--delta"), LVLA: css("--st-N3") }[q] || css("--muted");
  }

  // ---- canvas helpers
  function fitCanvas(c) {
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function nearestSnippet() {
    const t = state.tape;
    const key = t.kind === "sleep" ? "epoch" : "window";
    let best = t.snippets[0], d = Infinity;
    for (const s of t.snippets) {
      const dd = Math.abs(s[key] - state.pos);
      if (dd < d) { d = dd; best = s; }
    }
    return { snip: best, exact: d === 0 };
  }

  // ---- scope: the one loud element
  function renderScope(now = performance.now()) {
    const t = state.tape;
    if (!t) return;
    const { ctx, w, h } = fitCanvas($("scope"));
    ctx.clearRect(0, 0, w, h);
    const { snip } = nearestSnippet();
    if (!snip) return;
    const fs = snip.fs;
    const visible = Math.round((t.kind === "sleep" ? 4 : 2) * fs);
    const chans = Object.keys(snip.signals);
    const total = snip.signals[chans[0]].length;
    const start = Math.floor(state.scroll) % Math.max(1, total - visible);
    const artifact = t.kind === "sleep" ? t.epochs.artifact[state.pos] : t.windows.artifact[state.pos];

    // grid: one line per second
    ctx.strokeStyle = css("--grid");
    ctx.lineWidth = 1;
    const pxPerSample = w / visible;
    for (let s = Math.ceil(start / fs) * fs; s < start + visible; s += fs) {
      const x = (s - start) * pxPerSample;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }

    const trace = (arr, y0, scale, color, width = 1.5, alpha = 1) => {
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (let i = 0; i < visible; i++) {
        const v = arr[(start + i) % arr.length] || 0;
        const x = i * pxPerSample;
        const y = y0 - v * scale;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    };

    const legend = [];
    const uvScale = (lane) => lane / (t.kind === "sleep" ? 160 : 60);   // µV range per lane
    if (state.layer === "raw") {
      const lane = h / chans.length;
      chans.forEach((ch, k) => {
        trace(snip.signals[ch], lane * (k + 0.5), uvScale(lane), k ? css("--muted") : css("--ink"), 1.4);
        legend.push(`<span><i style="background:${k ? css("--muted") : css("--ink")}"></i>${ch.replace("_", "–")}</span>`);
      });
    } else if (state.layer === "bands") {
      const bands = Object.keys(snip.bands);
      const lane = h / bands.length;
      bands.forEach((b, k) => {
        trace(snip.bands[b], lane * (k + 0.5), uvScale(lane) * 1.4, bandColor(b), 1.6);
        legend.push(`<span><i style="background:${bandColor(b)}"></i>${BAND_NAMES[b]}</span>`);
      });
    } else {
      // model reading: faint raw signal + the band mix the model is looking at
      trace(snip.signals[chans[0]], h / 2, uvScale(h / 2) * 0.6, css("--ink"), 1.2, 0.25);
      const rel = t.kind === "sleep" ? t.epochs.rel : t.windows.rel;
      const bands = Object.keys(rel);
      const bw = Math.min(64, (w * 0.5) / bands.length);
      const x0 = w - bands.length * (bw + 10) - 16;
      bands.forEach((b, k) => {
        const v = rel[b][state.pos] || 0;
        const bh = Math.max(2, v * (h - 60));
        const x = x0 + k * (bw + 10);
        ctx.fillStyle = bandColor(b);
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x, h - 34 - bh, bw, bh);
        ctx.globalAlpha = 1;
        ctx.fillStyle = css("--muted");
        ctx.font = `12px ${css("--body")}`;
        ctx.textAlign = "center";
        ctx.fillText(`${BAND_NAMES[b].split(" ")[0]} ${Math.round(v * 100)}%`, x + bw / 2, h - 16);
      });
      legend.push(`<span>Share of signal power in each band, this window</span>`);
    }

    if (artifact) glitch(ctx, w, h, now);
    $("scope-legend").innerHTML = artifact
      ? `<span>Flagged: amplitude spike or flat line — the model skips this window</span>`
      : legend.join("");
  }

  function glitch(ctx, w, h, now) {
    const c = ctx.canvas;
    const dpr = window.devicePixelRatio || 1;
    const seed = reduceMotion ? 1 : Math.floor(now / 90);
    const rand = (k) => { const x = Math.sin(seed * 999 + k * 77) * 10000; return x - Math.floor(x); };
    for (let k = 0; k < 7; k++) {
      const y = rand(k) * h, sh = 4 + rand(k + 9) * 22, dx = (rand(k + 3) - 0.5) * 60;
      ctx.drawImage(c, 0, y * dpr, w * dpr, sh * dpr, dx, y, w, sh);
    }
    ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = css("--delta"); ctx.globalAlpha = 0.07; ctx.fillRect(3, 0, w, h);
    ctx.fillStyle = css("--alpha"); ctx.fillRect(-3, 0, w, h);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }

  function frame(now) {
    if (state.tape && !reduceMotion) {
      const dt = Math.min(100, now - (state.lastFrame || now));
      const snip = nearestSnippet().snip;
      if (snip) state.scroll += (dt / 1000) * snip.fs;   // real-time scroll
      renderScope(now);
    }
    state.lastFrame = now;
    requestAnimationFrame(frame);
  }

  // ---- timeline
  function renderTimeline() {
    state.tape.kind === "sleep" ? renderHypnogram() : renderCircumplex();
  }

  function renderHypnogram() {
    const t = state.tape, e = t.epochs;
    const { ctx, w, h } = fitCanvas($("hypnogram"));
    ctx.clearRect(0, 0, w, h);
    const n = e.t.length;
    const left = 44, ribbonH = 34, top = 10;
    const plotH = h - ribbonH - top - 18;
    const rowH = plotH / STAGE_ROWS.length;
    const x = (i) => left + (i / n) * (w - left);
    const rowY = (s) => top + STAGE_ROWS.indexOf(s) * rowH + rowH / 2;

    ctx.font = `12px ${css("--body")}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    STAGE_ROWS.forEach((s) => {
      ctx.fillStyle = css("--muted");
      ctx.fillText(s, left - 10, rowY(s));
      ctx.strokeStyle = css("--grid");
      ctx.beginPath(); ctx.moveTo(left, rowY(s)); ctx.lineTo(w, rowY(s)); ctx.stroke();
    });

    // model: coloured blocks on the predicted row
    const bw = Math.max(1, (w - left) / n);
    for (let i = 0; i < n; i++) {
      const p = predAt(i);
      if (p < 0 || e.artifact[i]) continue;
      const s = t.stages[p];
      ctx.fillStyle = stageColor(s);
      ctx.globalAlpha = 0.35 + 0.6 * (e.conf[i] || 0);
      ctx.fillRect(x(i), rowY(s) - rowH * 0.32, bw + 0.5, rowH * 0.64);
    }
    ctx.globalAlpha = 1;

    // expert: grey step line
    ctx.strokeStyle = css("--ink");
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const y = rowY(t.stages[e.true[i]]);
      i ? (ctx.lineTo(x(i), y), ctx.lineTo(x(i + 1), y)) : (ctx.moveTo(x(0), y), ctx.lineTo(x(1), y));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // artifacts: ticks above the plot
    ctx.fillStyle = css("--muted");
    for (let i = 0; i < n; i++) if (e.artifact[i]) ctx.fillRect(x(i), 0, Math.max(1, bw), 5);

    // spectral ribbon: stacked relative band power across the night
    const bands = Object.keys(e.rel);
    const ry = h - ribbonH - 14;
    for (let i = 0; i < n; i++) {
      let acc = 0;
      const tot = bands.reduce((s, b) => s + (e.rel[b][i] || 0), 0) || 1;
      for (const b of bands) {
        const v = (e.rel[b][i] || 0) / tot;
        ctx.fillStyle = bandColor(b);
        ctx.fillRect(x(i), ry + acc * ribbonH, bw + 0.5, v * ribbonH + 0.5);
        acc += v;
      }
    }
    ctx.fillStyle = css("--muted");
    ctx.textAlign = "right";
    ctx.fillText("bands", left - 10, ry + ribbonH / 2);

    // hour ticks
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    const minutes = e.t[n - 1];
    for (let m = 60; m < minutes; m += 60) {
      const i = e.t.findIndex((v) => v >= m);
      if (i > 0) ctx.fillText(`${m / 60}h`, x(i), h - 1);
    }

    // playhead
    ctx.strokeStyle = css("--state-color") || css("--ink");
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x(state.pos), 0); ctx.lineTo(x(state.pos), h - 14); ctx.stroke();

    // agreement for this night
    let agree = 0, scored = 0;
    for (let i = 0; i < n; i++) {
      const p = predAt(i);
      if (p < 0 || e.artifact[i]) continue;
      scored++;
      if (p === e.true[i]) agree++;
    }
    $("timeline-title").textContent = scored
      ? `The night, scored twice · ${Math.round((agree / scored) * 100)}% agreement`
      : "The night, scored twice";
  }

  function renderCircumplex() {
    const t = state.tape, wv = t.windows;
    const S = 360, pad = 36, c = S / 2, R = c - pad;
    const px = (p) => c + (p * 2 - 1) * R;          // probability -> position
    const py = (p) => c - (p * 2 - 1) * R;
    const rx = (r) => c + (((r - 1) / 8) * 2 - 1) * R; // 1–9 rating -> position
    const ry = (r) => c - (((r - 1) / 8) * 2 - 1) * R;
    // Trail is an exponential moving average (≈5 windows): single 4 s windows are noisy,
    // and the raw path reads as scribble. The live dot shows the raw current window.
    const pts = [];
    let ev = null, ea = null;
    for (let i = 0; i <= state.pos; i++) {
      if (wv.p_valence[i] == null) continue;
      ev = ev == null ? wv.p_valence[i] : 0.7 * ev + 0.3 * wv.p_valence[i];
      ea = ea == null ? wv.p_arousal[i] : 0.7 * ea + 0.3 * wv.p_arousal[i];
      pts.push([px(ev), py(ea)]);
    }
    const raw = wv.p_valence[state.pos] != null ? [px(wv.p_valence[state.pos]), py(wv.p_arousal[state.pos])] : null;
    const cur = raw;
    const ink = css("--ink"), faint = css("--faint"), bg = css("--bg");
    const now = css("--state-color") || ink;
    const trail = pts.map((p, i) => {
      const a = 0.15 + 0.6 * (i / Math.max(1, pts.length - 1));
      return `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="2.6" fill="${ink}" opacity="${a.toFixed(2)}"/>`;
    }).join("");
    const path = pts.length > 1 ? `<polyline points="${pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ")}" fill="none" stroke="${faint}" stroke-width="1"/>` : "";
    const done = state.pos >= wv.t.length - 1;
    const pred = t.prediction;
    const star = (x, y, r = 9) => {
      let d = "";
      for (let k = 0; k < 10; k++) {
        const rr = k % 2 ? r * 0.45 : r, a = (Math.PI / 5) * k - Math.PI / 2;
        d += `${k ? "L" : "M"}${(x + rr * Math.cos(a)).toFixed(1)},${(y + rr * Math.sin(a)).toFixed(1)}`;
      }
      return `<path d="${d}Z" fill="${quadColor(t.rating.quadrant)}" stroke="${css("--bg")}" stroke-width="1"/>`;
    };
    const ql = (x, y, q, anchor) =>
      `<text class="qlabel" x="${x}" y="${y}" text-anchor="${anchor}"><tspan font-weight="600">${QUAD[q].word}</tspan> (${QUAD[q].plain})</text>`;
    $("circumplex-wrap").innerHTML = `
      <svg viewBox="0 0 ${S} ${S}" role="img" aria-label="Valence and arousal reading over the song">
        <circle class="ring" cx="${c}" cy="${c}" r="${R}"/>
        <circle class="ring" cx="${c}" cy="${c}" r="${R / 2}"/>
        <line class="axis" x1="${pad}" y1="${c}" x2="${S - pad}" y2="${c}"/>
        <line class="axis" x1="${c}" y1="${pad}" x2="${c}" y2="${S - pad}"/>
        <text class="qlabel" x="${S - 4}" y="${c + 4}" text-anchor="end">pleasant</text>
        <text class="qlabel" x="4" y="${c + 4}">unpleasant</text>
        <text class="qlabel" x="${c + 6}" y="${pad + 4}">intense</text>
        <text class="qlabel" x="${c + 6}" y="${S - pad}">calm</text>
        ${ql(S - 8, 18, "HVHA", "end")}${ql(8, 18, "LVHA", "start")}
        ${ql(S - 8, S - 8, "HVLA", "end")}${ql(8, S - 8, "LVLA", "start")}
        ${path}${trail}
        ${cur ? `<circle cx="${cur[0].toFixed(1)}" cy="${cur[1].toFixed(1)}" r="7" fill="${now}" stroke="${bg}" stroke-width="2"/>` : ""}
        ${done ? `<circle cx="${px(pred.p_valence).toFixed(1)}" cy="${py(pred.p_arousal).toFixed(1)}" r="12" fill="none" stroke="${ink}" stroke-width="2"/>` : ""}
        ${star(rx(t.rating.valence), ry(t.rating.arousal))}
      </svg>`;
  }

  // ---- radar
  function renderRadar() {
    const rd = state.tape.radar;
    const W = 260, H = 240, cx = W / 2, cy = H / 2 + 4, R = 82;
    const n = rd.axes.length;
    const ang = (i) => -Math.PI / 2 + (i / n) * Math.PI * 2;
    const pt = (i, v) => [cx + Math.cos(ang(i)) * R * v, cy + Math.sin(ang(i)) * R * v];
    const poly = (vals) => vals.map((v, i) => pt(i, Math.max(0.03, v || 0)).map((z) => z.toFixed(1)).join(",")).join(" ");
    let web = "";
    for (const lvl of [0.25, 0.5, 0.75, 1]) web += `<polygon class="web" points="${poly(Array(n).fill(lvl))}"/>`;
    for (let i = 0; i < n; i++) {
      const [x, y] = pt(i, 1);
      web += `<line class="web" x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"/>`;
      const [lx, ly] = pt(i, 1.22);
      web += `<text x="${lx.toFixed(1)}" y="${(ly + 4).toFixed(1)}" text-anchor="middle">${BAND_NAMES[rd.axes[i]]}</text>`;
    }
    $("radar").innerHTML = `${web}<polygon class="power" points="${poly(rd.power)}"/><polygon class="sync" points="${poly(rd.sync)}"/>`;
    const who = state.tape.kind === "sleep" ? "the two scalp sites (forehead and back of head)" : "left and right sides of the head";
    $("radar-hint").innerHTML =
      `<span style="color:var(--theta)">Gold</span>: share of power in each band, against the group. ` +
      `<span style="color:var(--alpha)">Cyan</span>: how closely ${who} move together at that rhythm.`;
  }

  // ---- metrics
  function renderMetrics() {
    const m = state.metrics;
    const el = $("metrics");
    if (!m) { el.innerHTML = `<p class="hint">metrics.json not found.</p>`; return; }
    if (state.tape.kind === "sleep" && m.sleep) {
      const raw = m.sleep.raw, sm = m.sleep.smoothed;
      const conf = sm.confusion.map((row) => { const s = row.reduce((a, b) => a + b, 0) || 1; return row.map((v) => v / s); });
      const cells = [`<div class="hd"></div>`, ...m.sleep.classes.map((c) => `<div class="hd">${c}</div>`)];
      conf.forEach((row, i) => {
        cells.push(`<div class="hd">${m.sleep.classes[i]}</div>`);
        row.forEach((v, j) => cells.push(
          `<div style="background:color-mix(in srgb, ${stageColor(m.sleep.classes[j])} ${Math.round(v * 85)}%, transparent)">${Math.round(v * 100)}</div>`));
      });
      el.innerHTML =
        metric("Agreement beyond chance (κ)", sm.kappa, `${fmt(raw.kappa)} before smoothing. Expert scorers agree with each other around 0.75–0.8.`, sm.kappa, null) +
        metric("Balanced accuracy", sm.balanced_accuracy, "Every stage counts equally, so rare N1 isn't drowned out by N2.", sm.balanced_accuracy, 0.2) +
        `<div class="metric"><span class="name">Expert stage (rows) → model (columns), %</span><div class="note">${m.sleep.n_subjects} people, ${m.sleep.n_epochs.toLocaleString()} epochs, scored by models that never saw that person.</div><div class="confusion" style="grid-column:1/-1">${cells.join("")}</div></div>`;
    } else if (state.tape.kind === "emotion" && m.emotion) {
      const cs = m.emotion.cross_subject, ws = m.emotion.within_subject, nl = m.emotion.null || {};
      let html = "";
      for (const target of ["valence", "arousal"]) {
        const label = target === "valence" ? "Pleasant or not" : "Intense or calm";
        const chance = nl[target]?.p95 ?? 0.5;
        const chanceNote = nl[target] ? `shuffled-label 95th percentile ${fmt(chance)}` : "chance is 0.50";
        html += metric(`${label}, a stranger's songs`, cs[target].song.balanced_accuracy,
          `Song-level balanced accuracy, cross-subject. ${chanceNote}.`, cs[target].song.balanced_accuracy, chance);
        html += metric(`${label}, a known listener`, ws[target].song.balanced_accuracy,
          "Same people in training, unseen songs.", ws[target].song.balanced_accuracy, chance);
      }
      html += `<p class="hint">One rating per song, given afterwards — every 2 s window inherits it. That's why this is harder than sleep: the label is softer than the signal.</p>`;
      el.innerHTML = html;
    } else {
      el.innerHTML = `<p class="hint">No metrics for this kind of tape yet.</p>`;
    }
  }

  const fmt = (v) => (v == null ? "—" : Number(v).toFixed(2));
  function metric(name, value, note, barValue, chance) {
    const bar = barValue == null ? "" :
      `<div class="bar"><span style="width:${Math.max(0, Math.min(1, barValue)) * 100}%"></span>${chance != null ? `<i class="chance" style="left:${chance * 100}%" title="chance"></i>` : ""}</div>`;
    return `<div class="metric"><span class="name">${name}</span><span class="val">${fmt(value)}</span>${bar}<span class="note">${note}</span></div>`;
  }

  init();
})();
