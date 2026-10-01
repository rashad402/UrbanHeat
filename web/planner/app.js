/* UrbanHeat Planner — front end.
 *
 * Map: Mapbox GL JS satellite basemap + the Landsat thermal layer served as Earth Engine raster
 * tiles, so both pan and zoom at any scale. Ward boundaries are a vector source, which gives
 * precise hit-testing and GPU-side hover/selection styling via feature-state.
 *
 * The Mapbox token is a public (pk.) token delivered by /api/meta, which reads it from a
 * gitignored .env — it is never committed.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const api = (p, opts) => fetch(p, opts).then(r => {
    if (!r.ok) return r.json().then(e => { throw new Error(e.detail || r.statusText); });
    return r.json();
  });

  const state = {
    map: null, meta: null, bounds: null,
    wards: new Map(),
    selected: new Set(),
    tool: "select",
    bbox: null,
    hovered: null,
    iv: { albedo: 0, ndvi: 0, roofShare: 0.5 },
    season: "all",
    rank: [],
    reqId: 0,
  };

  const fail = (html) => { $("stageLoading").innerHTML = html; $("stageLoading").hidden = false; };

  let toastTimer = null;
  function toast(msg, isError) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.toggle("is-error", !!isError);
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 6000 : 3000);
  }

  /* ───────────────── slider chrome ─────────────────
   * Each <input type=range> gets wrapped in the layered slider: a rail, an animated fill with a
   * travelling sheen, and a pulsing halo tracking the thumb. The input itself stays the control —
   * it keeps keyboard stepping, focus and screen-reader semantics, and the layers behind it are
   * positioned purely off the --p custom property (0–1) on the wrapper. Styles degrade to a plain
   * styled slider if this never runs, so an early boot failure is not a broken rail.
   */
  const repaints = [];

  function enhanceRanges() {
    document.querySelectorAll("input[type=range]").forEach((input) => {
      if (input.closest(".sld")) return;

      const wrap = document.createElement("div");
      wrap.className = "sld";
      input.replaceWith(wrap);
      wrap.innerHTML = `<span class="sld-rail"></span>
        <span class="sld-fill"><i class="sld-sheen"></i></span>
        <span class="sld-halo"></span>`;
      wrap.appendChild(input);

      const paint = () => {
        const min = +input.min || 0;
        const max = input.max === "" ? 1 : +input.max;
        const span = max - min;
        const f = span > 0 ? (+input.value - min) / span : 0;
        wrap.style.setProperty("--p", f.toFixed(4));
        // A slider sitting at its floor is not a live value, so it stops pulsing.
        wrap.classList.toggle("is-zero", +input.value <= min);
      };
      input.addEventListener("input", paint);
      repaints.push(paint);
      paint();
    });
  }

  // Presets and any other code path that sets .value directly fires no 'input' event.
  const repaintRanges = () => repaints.forEach((p) => p());

  const body = () => ({
    selection: selectionPayload(),
    interventions: {
      albedo_set: state.iv.albedo, ndvi_delta: state.iv.ndvi, roof_share: state.iv.roofShare,
    },
    season: state.season,
  });

  /* ───────────────────────── boot ───────────────────────── */
  async function boot() {
    // Before anything that can fail: the rail is visible while the map and data load.
    enhanceRanges();

    let meta, wards;
    try {
      [meta, wards] = await Promise.all([api("/api/meta"), api("/api/wards")]);
    } catch (err) {
      return fail(`<p style="color:#b83820;max-width:36ch">Could not reach the backend.<br><br>
        <span style="color:#6b7c78">Start it with<br>
        <code style="font-family:IBM Plex Mono,monospace">uvicorn api.planner:app --port 8080</code></span></p>`);
    }
    state.meta = meta;
    state.bounds = meta.bounds;

    const lam = (meta.model.match(/lambda([0-9.]+?)\.pt/) || [])[1];
    $("chipModel").textContent = lam ? `PINN · λ=${lam}` : meta.model;
    $("chipWards").textContent = meta.n_wards + " wards with data";
    $("legendBar").style.background = `linear-gradient(90deg, ${meta.palette.join(",")})`;
    $("legLo").textContent = meta.lst_range[0] + "°C";
    $("legHi").textContent = meta.lst_range[1] + "°C";

    if (!window.mapboxgl || !mapboxgl.supported()) {
      return fail(`<p style="color:#8a6410;max-width:40ch">This browser cannot run Mapbox GL
        (WebGL unavailable).<br><br><span style="color:#6b7c78">Enable hardware acceleration, or
        use the static-image build.</span></p>`);
    }
    if (!meta.mapbox_token) {
      return fail(`<p style="color:#8a6410;max-width:40ch">No Mapbox token configured.<br><br>
        <span style="color:#6b7c78">Add <code>MAPBOX_TOKEN=…</code> to <code>.env</code> and restart.</span></p>`);
    }

    initMap(meta, wards);
    wireUI();
  }

  /* ───────────────────────── map ───────────────────────── */
  function initMap(meta, wards) {
    mapboxgl.accessToken = meta.mapbox_token;
    const [w, s, e, n] = meta.bounds;

    const map = new mapboxgl.Map({
      container: "map",
      style: "mapbox://styles/mapbox/satellite-streets-v12",
      bounds: [[w, s], [e, n]],
      fitBoundsOptions: { padding: 24 },
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
    });
    state.map = map;
    map.addControl(new mapboxgl.AttributionControl({ compact: true }), "bottom-right");
    map.addControl(new mapboxgl.ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-right");
    map.touchZoomRotate.disableRotation();

    map.on("error", (ev) => {
      const m = ev && ev.error && ev.error.message || "";
      if (/access token|Unauthorized|401/i.test(m)) {
        fail(`<p style="color:#b83820;max-width:40ch">Mapbox rejected the access token.<br><br>
          <span style="color:#6b7c78">Check it is a public <code>pk.</code> token and that any URL
          restriction allows <code>localhost</code>.</span></p>`);
      } else {
        console.warn("map error:", m);
      }
    });

    // Add layers from whichever signal arrives first. Relying on a single 'load' event has
    // twice left the UI stuck behind its own loading overlay, so this is idempotent and also
    // polled — the overlay can never outlive a usable map.
    let layersAdded = false;
    const ready = () => {
      if (layersAdded || !map.isStyleLoaded()) return;
      layersAdded = true;
      addLayers(map, meta, wards);
      $("stageLoading").hidden = true;
    };
    map.on("load", ready);
    map.on("style.load", ready);
    map.on("idle", ready);
    const poll = setInterval(() => { ready(); if (layersAdded) clearInterval(poll); }, 300);
    setTimeout(() => clearInterval(poll), 30000);

    // Watchdog: never leave the planner staring at a spinner. If the basemap has not loaded,
    // say what is most likely wrong instead of hanging.
    setTimeout(() => {
      if (layersAdded) return;
      fail(`<p style="color:#8a6410;max-width:42ch">The Mapbox basemap did not load.<br><br>
        <span style="color:#6b7c78">Most likely one of:<br>
        &bull; the token is URL-restricted and does not allow <code>localhost</code><br>
        &bull; no network access to <code>api.mapbox.com</code><br>
        &bull; the token has no remaining map loads<br><br>
        Everything else (wards, model, analysis) is working — only the basemap is missing.</span></p>`);
    }, 12000);
  }

  function addLayers(map, meta, wards) {
    {
      // Thermal layer — Earth Engine raster tiles
      map.addSource("lst", { type: "raster", tiles: [meta.lst_tiles], tileSize: 256 });
      map.addLayer({
        id: "lst", type: "raster", source: "lst",
        paint: { "raster-opacity": +$("opacity").value, "raster-resampling": "nearest" },
      });

      // Ward vectors.
      // These colours are deliberately NOT the interface accent. The chrome accent (#0d6e5e) is
      // chosen for contrast against white cards and disappears against satellite imagery, so the
      // overlay keeps the bright teal and white that read over a photographic basemap.
      map.addSource("wards", { type: "geojson", data: wards, promoteId: "ward_id" });
      map.addLayer({
        id: "ward-fill", type: "fill", source: "wards",
        paint: {
          "fill-color": ["case",
            ["boolean", ["feature-state", "selected"], false], "#19c2a8",
            ["boolean", ["feature-state", "hover"], false], "#ffffff",
            "#ffffff"],
          "fill-opacity": ["case",
            ["boolean", ["feature-state", "selected"], false], 0.35,
            ["boolean", ["feature-state", "hover"], false], 0.14,
            0.0],
        },
      });
      map.addLayer({
        id: "ward-line", type: "line", source: "wards",
        paint: {
          "line-color": ["case",
            ["boolean", ["feature-state", "selected"], false], "#5eead4", "rgba(255,255,255,.45)"],
          "line-width": ["case",
            ["boolean", ["feature-state", "selected"], false], 2.2, 0.8],
        },
      });

      for (const f of wards.features) state.wards.set(f.properties.ward_id, f.properties);

      wireMapInteractions(map);
    }
  }

  function wireMapInteractions(map) {
    const canvas = map.getCanvas();

    map.on("mousemove", "ward-fill", (ev) => {
      if (state.tool !== "select" || !ev.features.length) return;
      const id = ev.features[0].id;
      if (state.hovered === id) return;
      if (state.hovered != null) map.setFeatureState({ source: "wards", id: state.hovered }, { hover: false });
      state.hovered = id;
      map.setFeatureState({ source: "wards", id }, { hover: true });
      canvas.style.cursor = state.wards.get(id)?.has_data ? "pointer" : "not-allowed";
    });
    map.on("mouseleave", "ward-fill", () => {
      if (state.hovered != null) map.setFeatureState({ source: "wards", id: state.hovered }, { hover: false });
      state.hovered = null; canvas.style.cursor = "";
    });

    map.on("click", "ward-fill", (ev) => {
      if (state.tool !== "select" || !ev.features.length) return;
      const id = ev.features[0].id;
      if (!state.wards.get(id)?.has_data) return;
      toggleWard(id);
    });

    // Popup on hover for quick context
    const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
    map.on("mousemove", "ward-fill", (ev) => {
      if (state.tool !== "select" || !ev.features.length) return popup.remove();
      const p = ev.features[0].properties;
      popup.setLngLat(ev.lngLat).setHTML(
        p.has_data
          ? `<strong>${p.ward_name}</strong><br>${p.lst_c}°C · ${Math.round(p.built_frac * 100)}% built-up`
          : `<strong>${p.ward_name}</strong><br>no land pixels`
      ).addTo(map);
    });
    map.on("mouseleave", "ward-fill", () => popup.remove());

    wireBoxDraw(map);
  }

  /* Box-draw: screen-space rubber band, converted to lng/lat on release. */
  function wireBoxDraw(map) {
    const canvas = map.getCanvasContainer();
    const box = $("boxDraw");
    let start = null;

    const pos = (e) => {
      const r = map.getContainer().getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };

    canvas.addEventListener("pointerdown", (e) => {
      if (state.tool !== "draw" || (e.pointerType === "mouse" && e.button !== 0)) return;
      e.preventDefault(); e.stopPropagation();
      map.dragPan.disable();
      start = pos(e);
      box.hidden = false;
      Object.assign(box.style, { left: start[0] + "px", top: start[1] + "px", width: "0px", height: "0px" });
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp, { once: true });
    }, true);

    function onMove(e) {
      if (!start) return;
      const p = pos(e);
      Object.assign(box.style, {
        left: Math.min(start[0], p[0]) + "px", top: Math.min(start[1], p[1]) + "px",
        width: Math.abs(p[0] - start[0]) + "px", height: Math.abs(p[1] - start[1]) + "px",
      });
    }

    function onUp(e) {
      window.removeEventListener("pointermove", onMove);
      map.dragPan.enable();
      if (!start) return;
      const p = pos(e);
      const dx = Math.abs(p[0] - start[0]), dy = Math.abs(p[1] - start[1]);
      start = null;
      if (dx < 6 || dy < 6) { box.hidden = true; return; }      // ignore stray clicks
      const r = box.getBoundingClientRect(), c = map.getContainer().getBoundingClientRect();
      const nw = map.unproject([r.left - c.left, r.top - c.top]);
      const se = map.unproject([r.right - c.left, r.bottom - c.top]);
      clearWardSelection();
      state.bbox = [Math.min(nw.lng, se.lng), Math.min(nw.lat, se.lat),
                    Math.max(nw.lng, se.lng), Math.max(nw.lat, se.lat)];
      renderSelection(); analyze();
    }
  }

  /* ─────────────────── selection ─────────────────── */
  function setSel(id, on) {
    // Guarded: if the basemap failed, selection and analysis must still work.
    const m = state.map;
    if (!m || !m.getSource || !m.getSource("wards")) return;
    m.setFeatureState({ source: "wards", id }, { selected: on });
  }
  function clearWardSelection() {
    state.selected.forEach(id => setSel(id, false));
    state.selected.clear();
  }

  function syncRankSelection() {
    document.querySelectorAll(".rank-row").forEach(b =>
      b.classList.toggle("is-selected", state.selected.has(b.dataset.id)));
  }

  function toggleWard(id) {
    if (state.bbox) { state.bbox = null; $("boxDraw").hidden = true; }
    if (state.selected.has(id)) { state.selected.delete(id); setSel(id, false); }
    else { state.selected.add(id); setSel(id, true); }
    renderSelection(); syncRankSelection(); analyze();
  }

  function clearSelection() {
    clearWardSelection();
    state.bbox = null; $("boxDraw").hidden = true;
    renderSelection(); syncRankSelection(); resetResults();
    $("exportBtn").hidden = true;
  }

  function renderSelection() {
    const box = $("selectionBox");
    const n = state.selected.size;
    $("clearSel").hidden = !(n || state.bbox);

    if (state.bbox) {
      const [w, s, e, nn] = state.bbox;
      const km = (a, b) => Math.abs(a - b);
      box.innerHTML = `<div class="sel-summary"><span class="sel-count">1</span>
        <span class="sel-unit">custom area</span>
        <span class="sel-area">${(km(w, e) * 109.6).toFixed(2)} × ${(km(s, nn) * 110.6).toFixed(2)} km</span></div>
        <p class="empty" style="margin-top:6px"><span>Sampled live from Earth Engine at 30 m.</span></p>`;
      return;
    }
    if (!n) {
      box.innerHTML = `<p class="empty">No area selected.<br><span>Click a ward on the map, or use
        <strong>Draw area</strong> for a custom zone.</span></p>`;
      return;
    }
    let area = 0;
    const tags = [...state.selected].map(id => {
      const p = state.wards.get(id);
      area += p.area_sqkm || 0;
      return `<span class="tagx">${p.ward_name}<button data-rm="${id}" aria-label="Remove">×</button></span>`;
    }).join("");
    box.innerHTML = `<div class="sel-summary">
        <span class="sel-count">${n}</span><span class="sel-unit">ward${n > 1 ? "s" : ""}</span>
        <span class="sel-area">${area.toFixed(2)} km²</span></div>
      <div class="sel-names">${tags}</div>`;
    box.querySelectorAll("[data-rm]").forEach(b =>
      b.addEventListener("click", () => toggleWard(b.dataset.rm)));
  }

  /* ─────────────────── analysis ─────────────────── */
  function selectionPayload() {
    if (state.bbox) return { kind: "bbox", bounds: state.bbox };
    if (state.selected.size) return { kind: "wards", ward_ids: [...state.selected] };
    return null;
  }

  function resetResults() {
    $("kDelta").textContent = "—"; $("kArea").textContent = "—";
    $("kTemp").textContent = "—"; $("kBest").textContent = "—";
    $("compare").hidden = true; $("breakdown").hidden = true;
  }

  let timer = null;
  function analyze() { clearTimeout(timer); timer = setTimeout(runAnalyze, 160); }

  async function runAnalyze() {
    const sel = selectionPayload();
    if (!sel) return resetResults();
    const myId = ++state.reqId;
    $("busyDot").hidden = false;
    try {
      const res = await api("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()),
      });
      if (myId !== state.reqId) return;
      renderResults(res);
      $("exportBtn").hidden = false;
      if (sel.kind === "wards") runCompare(sel, myId); else $("compare").hidden = true;
    } catch (err) {
      if (myId !== state.reqId) return;
      resetResults();
      toast(String(err.message || err), true);     // errors belong in a toast, not a KPI tile
    } finally {
      if (myId === state.reqId) $("busyDot").hidden = true;
    }
  }

  function renderResults(res) {
    const s = res.summary, none = s.no_change;
    $("kDelta").textContent = none ? "—" : s.mean_delta_t.toFixed(2) + " °C";
    // Honest uncertainty: the deployed model's held-out RMSE, shown alongside the estimate.
    $("kDeltaErr").textContent = (none || !s.model_rmse) ? "" : `model error ±${s.model_rmse} K`;
    $("kArea").textContent = s.area_sqkm ? s.area_sqkm.toFixed(2) + " km²" : "—";
    $("kTemp").textContent = none ? s.mean_t_base.toFixed(1) + " °C"
      : s.mean_t_base.toFixed(1) + " → " + (s.mean_t_base + s.mean_delta_t).toFixed(1);
    $("kBest").textContent = none ? "set an intervention"
      : `${s.best_name} (${s.best_delta_t.toFixed(2)} °C)`;

    const rows = res.items.filter(i => i.ward_id !== "area");
    if (rows.length > 1) {
      $("breakdownRows").innerHTML = rows.map(i => `
        <div class="bd-row"><span class="bd-name" title="${i.name}">${i.name}</span>
          <span class="bd-base">${i.t_base.toFixed(1)}°</span>
          <span class="bd-dt">${none ? "—" : i.delta_t.toFixed(2)}</span></div>`).join("");
      $("breakdown").hidden = false;
    } else $("breakdown").hidden = true;
  }

  const STRATEGIES = [
    { name: "Cool roofs", albedo_set: 0.50, ndvi_delta: 0 },
    { name: "Street trees", albedo_set: 0, ndvi_delta: 0.20 },
    { name: "Green roofs", albedo_set: 0, ndvi_delta: 0.30 },
    { name: "Combined", albedo_set: 0.50, ndvi_delta: 0.25 },
  ];

  async function runCompare(sel, myId) {
    try {
      const out = await Promise.all(STRATEGIES.map(st =>
        api("/api/analyze", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            selection: sel, season: state.season,
            interventions: { albedo_set: st.albedo_set, ndvi_delta: st.ndvi_delta,
                             roof_share: state.iv.roofShare },
          }),
        }).then(r => r.summary.mean_delta_t).catch(() => 0)));
      if (myId !== state.reqId) return;
      const worst = Math.min(...out, -0.01);
      $("compareRows").innerHTML = STRATEGIES.map((st, i) => `
        <div class="cmp-row">
          <span class="cmp-name">${st.name}</span>
          <span class="cmp-track"><span class="cmp-fill" style="width:${Math.max(2, out[i] / worst * 100).toFixed(0)}%"></span></span>
          <span class="cmp-val">${out[i].toFixed(2)}</span>
        </div>`).join("");
      $("compare").hidden = false;
    } catch { /* comparison is a nicety; never block the main result */ }
  }

  /* ─────────────────── UI wiring ─────────────────── */
  function wireUI() {
    document.querySelectorAll(".tool[data-tool]").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll(".tool[data-tool]").forEach(x => x.classList.remove("is-active"));
      b.classList.add("is-active");
      state.tool = b.dataset.tool;
      const drawing = state.tool === "draw";
      $("map").classList.toggle("is-drawing", drawing);
      $("mapHint").textContent = drawing
        ? "Drag on the map to sample a custom area"
        : "Click wards to build a scenario · drag to pan · scroll to zoom";
      if (state.map) drawing ? state.map.dragPan.disable() : state.map.dragPan.enable();
    }));

    $("zoomIn").addEventListener("click", () => state.map && state.map.zoomIn());
    $("zoomOut").addEventListener("click", () => state.map && state.map.zoomOut());
    $("zoomReset").addEventListener("click", () => {
      if (!state.map) return;
      const [w, s, e, n] = state.bounds;
      state.map.fitBounds([[w, s], [e, n]], { padding: 24 });
    });

    $("opacity").addEventListener("input", e => {
      if (state.map && state.map.getLayer("lst"))
        state.map.setPaintProperty("lst", "raster-opacity", +e.target.value);
    });

    const sync = () => {
      $("albedoOut").textContent = state.iv.albedo ? "α → " + state.iv.albedo.toFixed(2) : "off";
      $("ndviOut").textContent = state.iv.ndvi ? "+" + state.iv.ndvi.toFixed(2) + " NDVI" : "off";
      document.querySelectorAll(".preset").forEach(p => p.classList.toggle("is-active",
        Math.abs(+p.dataset.albedo - state.iv.albedo) < 1e-9 && Math.abs(+p.dataset.ndvi - state.iv.ndvi) < 1e-9));
      repaintRanges();
    };
    $("albedo").addEventListener("input", e => { state.iv.albedo = +e.target.value; sync(); analyze(); });
    $("ndvi").addEventListener("input", e => { state.iv.ndvi = +e.target.value; sync(); analyze(); });
    document.querySelectorAll(".preset").forEach(p => p.addEventListener("click", () => {
      state.iv.albedo = +p.dataset.albedo; state.iv.ndvi = +p.dataset.ndvi;
      $("albedo").value = state.iv.albedo; $("ndvi").value = state.iv.ndvi;
      sync(); analyze();
    }));
    sync();

    // season
    document.querySelectorAll(".seg").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll(".seg").forEach(x => {
        x.classList.remove("is-active"); x.setAttribute("aria-checked", "false");
      });
      b.classList.add("is-active"); b.setAttribute("aria-checked", "true");
      state.season = b.dataset.season;
      analyze(); if (state.rank.length) loadRank();
    }));

    // coverage assumption
    $("roofShare").addEventListener("input", e => {
      state.iv.roofShare = +e.target.value;
      $("roofShareOut").textContent = Math.round(state.iv.roofShare * 100) + "%";
      analyze();
    });

    $("rankRefresh").addEventListener("click", loadRank);
    $("exportBtn").addEventListener("click", exportCsv);
    $("clearSel").addEventListener("click", clearSelection);
    $("aboutBtn").addEventListener("click", () => $("aboutDlg").showModal());
  }

  /* ─────────────────── prioritisation ─────────────────── */
  async function loadRank() {
    $("rankList").innerHTML = `<p class="rank-empty">Scoring all wards…</p>`;
    try {
      const res = await api("/api/rank", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selection: { kind: "wards" }, season: state.season,
          interventions: { albedo_set: state.iv.albedo || 0.5, ndvi_delta: state.iv.ndvi,
                           roof_share: state.iv.roofShare },
        }),
      });
      state.rank = res.items.sort((a, b) => a.delta_t - b.delta_t);
      renderRank();
    } catch (err) {
      $("rankList").innerHTML = `<p class="rank-empty">Could not rank wards.</p>`;
      toast(String(err.message || err), true);
    }
  }

  function renderRank() {
    if (!state.rank.length) {
      $("rankList").innerHTML = `<p class="rank-empty">Press “Rank all”.</p>`;
      return;
    }
    $("rankList").innerHTML = state.rank.map((w, i) => `
      <button class="rank-row${state.selected.has(w.ward_id) ? " is-selected" : ""}"
              role="listitem" data-id="${w.ward_id}"
              title="${w.name} — baseline ${w.t_base}°C, ${Math.round(w.built_frac * 100)}% built-up">
        <span class="rank-n">${i + 1}</span>
        <span class="rank-name">${w.name}</span>
        <span class="rank-base">${w.t_base.toFixed(1)}°</span>
        <span class="rank-dt">${w.delta_t.toFixed(2)}</span>
      </button>`).join("");
    $("rankList").querySelectorAll(".rank-row").forEach(b =>
      b.addEventListener("click", () => toggleWard(b.dataset.id)));
  }

  async function exportCsv() {
    if (!selectionPayload()) return toast("Select an area first.");
    try {
      const r = await fetch("/api/export", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()),
      });
      if (!r.ok) throw new Error(await r.text());
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `urbanheat-scenario-${state.season}.csv`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      toast("Scenario exported.");
    } catch (err) { toast(String(err.message || err), true); }
  }

  boot();
})();
