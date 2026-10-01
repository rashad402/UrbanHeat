/* UrbanHeat Planner — front end.
 *
 * Map: Mapbox GL JS satellite basemap + the Landsat thermal layer served as Earth Engine raster
 * tiles, so both pan and zoom at any scale. Ward boundaries are a vector source, which gives
 * precise hit-testing and GPU-side hover/selection styling via feature-state.
 *
 * The Mapbox token is a public (pk.) token delivered by /api/meta, which reads it from a
 * gitignored .env — it is never committed.
 *
 * THREE THINGS WORTH KNOWING BEFORE EDITING
 *   1. Earth Engine tile URLs are signed and EXPIRE. When they do, the thermal layer silently
 *      stops drawing — no exception, no console error that distinguishes it from a slow tile. The
 *      map's error handler counts tile failures and asks /api/refresh_tiles to re-sign, rather
 *      than leaving a planner looking at a basemap and assuming the city has no heat data.
 *   2. Selection must not be mouse-only. Clicking the map is one route; the ward list in the rail
 *      is the keyboard and screen-reader route to exactly the same state, and both go through
 *      toggleWard so they can never diverge.
 *   3. Scenarios live in localStorage and every read and write is wrapped — private windows and
 *      blocked site data make these throw, and a planner losing the app because storage is
 *      unavailable would be worse than losing the saved scenarios.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const api = (p, opts) => fetch(p, opts).then(r => {
    if (!r.ok) return r.json().then(e => { throw new Error(e.detail || r.statusText); });
    return r.json();
  });

  const STORE_KEY = "urbanheat.scenarios.v1";
  const COACH_KEY = "urbanheat.coachSeen.v1";
  const MAX_PICK = 2;

  const state = {
    map: null, meta: null, bounds: null,
    wards: new Map(),
    selected: new Set(),
    tool: "select",
    bbox: null,
    poly: null,            // committed polygon ring, [[lon,lat], ...]
    tracing: [],           // vertices while tracing
    hovered: null,
    // roofShare stays null until the planner moves the slider. Sending a value on every request
    // would override the MEASURED per-ward roof shares with one flat city-wide number, silently
    // throwing away the footprint measurement the backend went to the trouble of making.
    iv: { albedo: 0, ndvi: 0, roofShare: null },
    roofShareDefault: 0.5,
    season: "all",
    rank: [],
    rankBy: "delta",
    choropleth: "none",
    scenarios: [],
    picked: [],
    cursor: -1,            // keyboard cursor into the filtered ward list
    filter: "",
    reqId: 0,
    lastSummary: null,
    tileFails: 0,
    refreshing: false,
  };

  const fail = (html) => { $("stageLoading").innerHTML = html; $("stageLoading").hidden = false; };
  const announce = (msg) => { $("liveRegion").textContent = msg; };

  let toastTimer = null;
  function toast(msg, isError) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.toggle("is-error", !!isError);
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 6000 : 3000);
  }

  const body = () => ({
    selection: selectionPayload(),
    interventions: {
      albedo_set: state.iv.albedo, ndvi_delta: state.iv.ndvi, roof_share: state.iv.roofShare,
    },
    season: state.season,
  });

  /* ───────────────────────── boot ───────────────────────── */
  async function boot() {
    let meta, wards;
    try {
      [meta, wards] = await Promise.all([api("/api/meta"), api("/api/wards")]);
    } catch (err) {
      return fail(`<p style="color:#f06a5d;max-width:36ch">Could not reach the backend.<br><br>
        <span style="color:#8a8498">Start it with<br>
        <code style="font-family:IBM Plex Mono,monospace">uvicorn api.planner:app --port 8080</code></span></p>`);
    }
    state.meta = meta;
    state.bounds = meta.bounds;
    state.roofShareDefault = typeof meta.roof_share_default === "number"
      ? meta.roof_share_default : 0.5;

    const lam = (meta.model.match(/lambda([0-9.]+?)\.pt/) || [])[1];
    $("chipModel").textContent = lam ? `PINN · λ=${lam}` : meta.model;
    $("chipWards").textContent = meta.n_wards + " wards with data";
    $("legendBar").style.background = `linear-gradient(90deg, ${meta.palette.join(",")})`;
    $("legLo").textContent = meta.lst_range[0] + "°C";
    $("legHi").textContent = meta.lst_range[1] + "°C";

    renderVintage(meta);
    state.wardsGeo = wards;
    for (const f of wards.features) state.wards.set(f.properties.ward_id, f.properties);
    state.scenarios = loadScenarios();
    renderWardList();
    renderScenarios();
    wireUI();
    maybeCoach();

    if (!window.mapboxgl || !mapboxgl.supported()) {
      return fail(`<p style="color:#e3a94e;max-width:40ch">This browser cannot run Mapbox GL
        (WebGL unavailable).<br><br><span style="color:#8a8498">The ward list and all analysis
        still work — only the map is unavailable.</span></p>`);
    }
    if (!meta.mapbox_token) {
      return fail(`<p style="color:#e3a94e;max-width:40ch">No Mapbox token configured.<br><br>
        <span style="color:#8a8498">Add <code>MAPBOX_TOKEN=…</code> to <code>.env</code> and restart.</span></p>`);
    }

    initMap(meta, wards);
  }

  function renderVintage(meta) {
    const v = meta.vintage || {};
    const win = v.first_date && v.last_date ? `${v.first_date} → ${v.last_date}`
      : `${v.start || ""}–${v.end || ""}`;
    $("statVintage").innerHTML =
      `<b>Baseline</b> ${v.n_scenes ? v.n_scenes + " scenes" : "Landsat 8/9"}, ${win}`;
    $("statModel").innerHTML = `<b>Model</b> ${meta.model_rmse
      ? `RMSE ${meta.model_rmse} K` : "—"}${meta.model_r2 ? ` · R² ${meta.model_r2}` : ""}`;

    const facts = [
      ["Sensor", v.sensor || "Landsat 8/9 C2 L2"],
      ["Scenes", v.n_scenes ?? "—"],
      ["Window", win],
      ["Pixels", (v.n_pixels || 0).toLocaleString()],
      ["Model", meta.model],
      ["Held-out RMSE", meta.model_rmse ? meta.model_rmse + " K" : "—"],
      ["Held-out R²", meta.model_r2 ?? "—"],
      ["Roof share", meta.roof_share_measured
        ? `measured (city ${meta.roof_share_default})`
        : `assumed ${meta.roof_share_default}`],
    ];
    $("dlgFacts").innerHTML = facts
      .map(([k, val]) => `<dt>${k}</dt><dd>${val}</dd>`).join("");

    $("roofShare").value = state.roofShareDefault;
    renderRoofShare();
    const inert = meta.n_wards_no_data || 0;
    $("wardCount").textContent = inert
      ? `${meta.n_wards} with data · ${inert} without`
      : `${meta.n_wards} wards`;
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
      const m = (ev && ev.error && ev.error.message) || "";
      if (/access token|Unauthorized|401/i.test(m)) {
        fail(`<p style="color:#f06a5d;max-width:40ch">Mapbox rejected the access token.<br><br>
          <span style="color:#8a8498">Check it is a public <code>pk.</code> token and that any URL
          restriction allows <code>localhost</code>.</span></p>`);
        return;
      }
      // An expired Earth Engine signature surfaces as repeated tile failures on the lst source.
      const src = ev && ev.sourceId;
      if (src === "lst" || /earthengine|403|Forbidden/i.test(m)) {
        if (++state.tileFails >= 3) refreshTiles();
      }
      console.warn("map error:", m);
    });

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

    setTimeout(() => {
      if (layersAdded) return;
      fail(`<p style="color:#e3a94e;max-width:42ch">The Mapbox basemap did not load.<br><br>
        <span style="color:#8a8498">Most likely one of:<br>
        &bull; the token is URL-restricted and does not allow <code>localhost</code><br>
        &bull; no network access to <code>api.mapbox.com</code><br>
        &bull; the token has no remaining map loads<br><br>
        Everything else (wards, model, analysis) is working — only the basemap is missing.</span></p>`);
    }, 12000);

    // Re-sign the tile URL before it can expire, so a long planning session never loses the
    // thermal layer mid-use.
    const ttl = (meta.tile_ttl_s || 2700) * 1000;
    setInterval(refreshTiles, Math.max(ttl * 0.8, 300000));
  }

  async function refreshTiles() {
    if (state.refreshing || !state.map) return;
    state.refreshing = true;
    try {
      const { lst_tiles } = await api("/api/refresh_tiles", { method: "POST" });
      const map = state.map;
      if (map.getLayer("lst")) map.removeLayer("lst");
      if (map.getSource("lst")) map.removeSource("lst");
      map.addSource("lst", { type: "raster", tiles: [lst_tiles], tileSize: 256 });
      map.addLayer({
        id: "lst", type: "raster", source: "lst",
        paint: { "raster-opacity": +$("opacity").value, "raster-resampling": "nearest" },
      }, firstWardLayer(map));
      state.tileFails = 0;
    } catch (err) {
      console.warn("tile refresh failed:", err);
    } finally {
      state.refreshing = false;
    }
  }

  const firstWardLayer = (map) => (map.getLayer("ward-fill") ? "ward-fill" : undefined);

  function addLayers(map, meta, wards) {
    map.addSource("lst", { type: "raster", tiles: [meta.lst_tiles], tileSize: 256 });
    map.addLayer({
      id: "lst", type: "raster", source: "lst",
      paint: { "raster-opacity": +$("opacity").value, "raster-resampling": "nearest" },
    });

    map.addSource("wards", { type: "geojson", data: wards, promoteId: "ward_id" });
    map.addLayer({ id: "ward-fill", type: "fill", source: "wards", paint: fillPaint() });
    map.addLayer({
      id: "ward-line", type: "line", source: "wards",
      filter: ["==", ["get", "has_data"], true],
      paint: {
        "line-color": ["case",
          ["boolean", ["feature-state", "selected"], false], "#5eead4", "rgba(255,255,255,.45)"],
        "line-width": ["case",
          ["boolean", ["feature-state", "selected"], false], 2.2, 0.8],
      },
    });
    // Wards with no land pixels get their own dashed outline. 27 of Kochi's 74 wards are
    // water-dominated and cannot be analysed; leaving them looking identical to the rest reads
    // as a broken app, so they are marked as deliberately inert on the map itself rather than
    // only in a hover popup.
    map.addLayer({
      id: "ward-line-nodata", type: "line", source: "wards",
      filter: ["!=", ["get", "has_data"], true],
      paint: {
        "line-color": "rgba(255,255,255,.3)", "line-width": 0.9, "line-dasharray": [2, 2],
      },
    });

    // Drawn zone (polygon tool).
    map.addSource("draw", { type: "geojson", data: emptyFC() });
    map.addLayer({
      id: "draw-fill", type: "fill", source: "draw",
      filter: ["==", ["geometry-type"], "Polygon"],
      paint: { "fill-color": "#19c2a8", "fill-opacity": 0.18 },
    });
    map.addLayer({
      id: "draw-line", type: "line", source: "draw",
      paint: { "line-color": "#5eead4", "line-width": 1.8, "line-dasharray": [2, 1.5] },
    });
    map.addLayer({
      id: "draw-vertex", type: "circle", source: "draw",
      filter: ["==", ["geometry-type"], "Point"],
      paint: {
        "circle-radius": 4.5, "circle-color": "#06211d",
        "circle-stroke-color": "#5eead4", "circle-stroke-width": 1.8,
      },
    });

    wireMapInteractions(map);
    applyChoropleth();
  }

  const emptyFC = () => ({ type: "FeatureCollection", features: [] });

  /* ─────────────── choropleth ─────────────── */
  function fillPaint() {
    const mode = state.choropleth;
    const [lo, hi] = state.meta.lst_range;
    const pal = state.meta.palette;

    let color;
    if (mode === "lst") {
      // Same ramp as the raster legend, so a shaded ward and the pixels under it agree.
      const stops = pal.flatMap((c, i) => [lo + (hi - lo) * (i / (pal.length - 1)), c]);
      color = ["case",
        ["!=", ["get", "has_data"], true], "#000000",
        ["interpolate", ["linear"], ["coalesce", ["get", "lst_c"], lo], ...stops]];
    } else if (mode === "delta") {
      // Cooling is negative, so the ramp runs from 0 (no effect) to the strongest ΔT seen.
      const worst = Math.min(-0.05, ...state.rank.map(r => r.delta_t));
      // coalesce rather than an equality test against null: Mapbox expressions reject `null`
      // as an == operand, and an unranked ward should simply read as "no effect".
      color = ["case",
        ["!=", ["get", "has_data"], true], "#000000",
        ["interpolate", ["linear"], ["coalesce", ["feature-state", "dt"], 0],
          worst, "#5eead4", worst * 0.5, "#19c2a8", 0, "#1c1b21"]];
    } else {
      color = ["case",
        ["boolean", ["feature-state", "selected"], false], "#19c2a8",
        ["!=", ["get", "has_data"], true], "#000000",
        "#ffffff"];
    }

    const shaded = mode !== "none";
    return {
      "fill-color": color,
      "fill-opacity": ["case",
        ["!=", ["get", "has_data"], true], 0.3,
        ["boolean", ["feature-state", "selected"], false], shaded ? 0.85 : 0.35,
        ["boolean", ["feature-state", "hover"], false], shaded ? 0.72 : 0.14,
        shaded ? 0.56 : 0.0],
    };
  }

  function applyChoropleth() {
    const map = state.map;
    if (map && map.getLayer("ward-fill")) {
      const paint = fillPaint();
      for (const [k, v] of Object.entries(paint)) map.setPaintProperty("ward-fill", k, v);
    }
    // The explanatory note is set regardless: it must still be correct when the basemap failed
    // to load, since every panel keeps working without it.
    const notes = {
      none: "Ward colour comes from the raster. Choose a mode to shade wards by their own values.",
      lst: "Wards shaded by their measured mean land surface temperature.",
      delta: "Wards shaded by predicted cooling for the current scenario. Needs a ranking run.",
    };
    $("choroNote").textContent = notes[state.choropleth];
  }

  function pushRankToMap() {
    const map = state.map;
    if (!map || !map.getSource("wards")) return;
    for (const r of state.rank) {
      map.setFeatureState({ source: "wards", id: r.ward_id }, { dt: r.delta_t });
    }
    if (state.choropleth === "delta") applyChoropleth();
  }

  /* ─────────────── map interactions ─────────────── */
  function wireMapInteractions(map) {
    const canvas = map.getCanvas();
    const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });

    map.on("mousemove", "ward-fill", (ev) => {
      if (state.tool !== "select" || !ev.features.length) { popup.remove(); return; }
      const f = ev.features[0];
      const id = f.id;
      if (state.hovered !== id) {
        if (state.hovered != null)
          map.setFeatureState({ source: "wards", id: state.hovered }, { hover: false });
        state.hovered = id;
        map.setFeatureState({ source: "wards", id }, { hover: true });
        canvas.style.cursor = state.wards.get(id)?.has_data ? "pointer" : "not-allowed";
      }
      const p = f.properties;
      const ranked = state.rank.find(r => r.ward_id === id);
      popup.setLngLat(ev.lngLat).setHTML(
        p.has_data
          ? `<strong>${p.ward_name}</strong><br>${p.lst_c}°C · ${Math.round(p.built_frac * 100)}% built-up`
            + (ranked ? `<br>ΔT ${ranked.delta_t.toFixed(1)} °C` : "")
          : `<strong>${p.ward_name}</strong><br>no land pixels — cannot be analysed`
      ).addTo(map);
    });
    map.on("mouseleave", "ward-fill", () => {
      if (state.hovered != null)
        map.setFeatureState({ source: "wards", id: state.hovered }, { hover: false });
      state.hovered = null; canvas.style.cursor = ""; popup.remove();
    });

    map.on("click", "ward-fill", (ev) => {
      if (state.tool !== "select" || !ev.features.length) return;
      const id = ev.features[0].id;
      if (!state.wards.get(id)?.has_data) {
        toast(`${state.wards.get(id)?.ward_name || "That ward"} has no land pixels — water, cloud or outside the city.`);
        return;
      }
      toggleWard(id);
    });

    wireBoxDraw(map);
    wirePolygonDraw(map);
  }

  /* Box-draw: screen-space rubber band, converted to lng/lat on release.
     Pointer events rather than mouse events, so a stylus or finger drag works on a tablet — the
     device planners actually carry on site. */
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
      window.addEventListener("pointercancel", onUp, { once: true });
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
      if (dx < 6 || dy < 6) { box.hidden = true; return; }      // ignore stray taps
      const r = box.getBoundingClientRect(), c = map.getContainer().getBoundingClientRect();
      const nw = map.unproject([r.left - c.left, r.top - c.top]);
      const se = map.unproject([r.right - c.left, r.bottom - c.top]);
      clearWardSelection();
      state.poly = null;
      state.bbox = [Math.min(nw.lng, se.lng), Math.min(nw.lat, se.lat),
                    Math.max(nw.lng, se.lng), Math.max(nw.lat, se.lat)];
      renderSelection(); analyze();
    }
  }

  /* Polygon-draw: real planning zones are not rectangles. Tap or click each vertex; finish with
     the floating bar, Enter, or a tap on the first vertex. Pointer events again, so this is
     usable with a finger. */
  function wirePolygonDraw(map) {
    const canvas = map.getCanvasContainer();
    let down = null;

    canvas.addEventListener("pointerdown", (e) => {
      if (state.tool !== "polygon" || (e.pointerType === "mouse" && e.button !== 0)) return;
      down = { x: e.clientX, y: e.clientY };
    }, true);

    canvas.addEventListener("pointerup", (e) => {
      if (state.tool !== "polygon" || !down) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      down = null;
      if (moved > 8) return;                     // that was a pan, not a vertex
      e.preventDefault(); e.stopPropagation();
      const r = map.getContainer().getBoundingClientRect();
      const ll = map.unproject([e.clientX - r.left, e.clientY - r.top]);

      // Tapping the first vertex closes the ring.
      if (state.tracing.length >= 3) {
        const first = map.project(state.tracing[0]);
        if (Math.hypot(first.x - (e.clientX - r.left), first.y - (e.clientY - r.top)) < 14) {
          return finishPolygon();
        }
      }
      state.tracing.push([ll.lng, ll.lat]);
      renderTracing();
    }, true);

    map.on("dblclick", () => { if (state.tool === "polygon") finishPolygon(); });
  }

  /* Draw a ring on the map. `withVertices` is false for a committed zone, where the vertex
     handles would only be visual noise. */
  function drawRing(pts, withVertices) {
    const map = state.map;
    if (!map || !map.getSource("draw")) return;
    const feats = withVertices
      ? pts.map(c => ({ type: "Feature", geometry: { type: "Point", coordinates: c } }))
      : [];
    if (pts.length >= 2) {
      feats.push({ type: "Feature", geometry: { type: "LineString", coordinates: pts } });
    }
    if (pts.length >= 3) {
      feats.push({ type: "Feature",
        geometry: { type: "Polygon", coordinates: [[...pts, pts[0]]] } });
    }
    map.getSource("draw").setData({ type: "FeatureCollection", features: feats });
  }

  function renderTracing() {
    drawRing(state.tracing, true);
    traceBar(state.tracing.length);
  }

  function traceBar(n) {
    let bar = $("traceBar");
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "traceBar";
      bar.className = "hint hint-trace";
      $("stage").appendChild(bar);
    }
    bar.hidden = state.tool !== "polygon";
    bar.innerHTML = n === 0
      ? `Tap or click each corner of the zone`
      : `<span>${n} point${n > 1 ? "s" : ""}</span>
         <button class="link-btn" id="traceFinish" ${n < 3 ? "disabled" : ""}>Finish</button>
         <button class="link-btn" id="traceUndo">Undo</button>
         <button class="link-btn" id="traceCancel">Cancel</button>`;
    const fin = $("traceFinish"), un = $("traceUndo"), ca = $("traceCancel");
    if (fin) fin.onclick = finishPolygon;
    if (un) un.onclick = () => { state.tracing.pop(); renderTracing(); };
    if (ca) ca.onclick = cancelTracing;
  }

  function finishPolygon() {
    if (state.tracing.length < 3) return toast("A zone needs at least three points.");
    clearWardSelection();
    state.bbox = null; $("boxDraw").hidden = true;
    state.poly = state.tracing.slice();
    state.tracing = [];
    drawRing(state.poly, false);
    traceBar(0);
    renderSelection(); analyze();
  }

  function cancelTracing() {
    state.tracing = [];
    renderTracing();
  }

  function clearDrawn() {
    state.poly = null; state.bbox = null; state.tracing = [];
    $("boxDraw").hidden = true;
    const map = state.map;
    if (map && map.getSource("draw")) map.getSource("draw").setData(emptyFC());
    traceBar(0);
  }

  /* ─────────────────── selection ─────────────────── */
  function setSel(id, on) {
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
    document.querySelectorAll(".ward-row").forEach(b => {
      const on = state.selected.has(b.dataset.id);
      b.classList.toggle("is-selected", on);
      b.setAttribute("aria-selected", String(on));
    });
  }

  function toggleWard(id) {
    const p = state.wards.get(id);
    if (!p || !p.has_data) return;
    clearDrawn();
    if (state.selected.has(id)) { state.selected.delete(id); setSel(id, false); }
    else { state.selected.add(id); setSel(id, true); }
    renderSelection(); syncRankSelection(); analyze();
    announce(`${p.ward_name} ${state.selected.has(id) ? "added to" : "removed from"} the selection.`
      + ` ${state.selected.size} ward${state.selected.size === 1 ? "" : "s"} selected.`);
  }

  function clearSelection() {
    clearWardSelection();
    clearDrawn();
    renderSelection(); syncRankSelection(); resetResults();
    $("exportBtn").hidden = true;
    announce("Selection cleared.");
  }

  function renderSelection() {
    const box = $("selectionBox");
    const n = state.selected.size;
    $("clearSel").hidden = !(n || state.bbox || state.poly);

    if (state.poly) {
      box.innerHTML = `<div class="sel-summary"><span class="sel-count">1</span>
        <span class="sel-unit">drawn zone</span>
        <span class="sel-area">${state.poly.length} points</span></div>
        <p class="empty" style="margin-top:6px"><span>Sampled live from Earth Engine at 30 m;
        area measured from the shape.</span></p>`;
      return;
    }
    if (state.bbox) {
      const [w, s, e, nn] = state.bbox;
      box.innerHTML = `<div class="sel-summary"><span class="sel-count">1</span>
        <span class="sel-unit">custom area</span>
        <span class="sel-area">${(Math.abs(w - e) * 109.6).toFixed(2)} × ${(Math.abs(s - nn) * 110.6).toFixed(2)} km</span></div>
        <p class="empty" style="margin-top:6px"><span>Sampled live from Earth Engine at 30 m.</span></p>`;
      return;
    }
    if (!n) {
      box.innerHTML = `<p class="empty">No area selected.<br><span>Click a ward on the map, pick one
        from the <strong>Wards</strong> list, or draw a <strong>Box</strong> or <strong>Zone</strong>.</span></p>`;
      return;
    }
    let area = 0;
    const tags = [...state.selected].map(id => {
      const p = state.wards.get(id);
      area += p.area_sqkm || 0;
      return `<span class="tagx">${p.ward_name}<button data-rm="${id}" aria-label="Remove ${p.ward_name}">×</button></span>`;
    }).join("");
    box.innerHTML = `<div class="sel-summary">
        <span class="sel-count">${n}</span><span class="sel-unit">ward${n > 1 ? "s" : ""}</span>
        <span class="sel-area">${area.toFixed(2)} km²</span></div>
      <div class="sel-names">${tags}</div>`;
    box.querySelectorAll("[data-rm]").forEach(b =>
      b.addEventListener("click", () => toggleWard(b.dataset.rm)));
  }

  /* ─────────────── keyboard-accessible ward list ─────────────── */
  function filteredWards() {
    const q = state.filter.trim().toLowerCase();
    return [...state.wards.values()]
      .filter(p => !q || (p.ward_name || "").toLowerCase().includes(q)
        || String(p.ward_no || "").includes(q))
      .sort((a, b) => (a.ward_no || 0) - (b.ward_no || 0));
  }

  function renderWardList() {
    const rows = filteredWards();
    const list = $("wardList");
    if (!rows.length) {
      list.innerHTML = `<p class="scen-empty" style="padding:9px">No ward matches that filter.</p>`;
      return;
    }
    list.innerHTML = rows.map((p, i) => `
      <button class="ward-row${state.selected.has(p.ward_id) ? " is-selected" : ""}${i === state.cursor ? " is-cursor" : ""}"
              role="option" aria-selected="${state.selected.has(p.ward_id)}"
              data-id="${p.ward_id}" data-i="${i}" tabindex="-1"
              ${p.has_data ? "" : 'aria-disabled="true"'}>
        <span class="ward-name">${p.ward_name}</span>
        <span class="ward-t">${p.has_data ? p.lst_c + "°" : ""}</span>
        <span class="ward-flag">${p.has_data ? "" : "no data"}</span>
      </button>`).join("");
    list.querySelectorAll(".ward-row").forEach(b => b.addEventListener("click", () => {
      state.cursor = +b.dataset.i;
      if (b.getAttribute("aria-disabled") === "true") {
        toast("That ward has no land pixels — water, cloud or outside the city.");
        return;
      }
      toggleWard(b.dataset.id);
      renderWardList();
    }));
  }

  function moveCursor(step) {
    const rows = filteredWards();
    if (!rows.length) return;
    state.cursor = Math.max(0, Math.min(rows.length - 1, state.cursor + step));
    renderWardList();
    const el = $("wardList").querySelector(".is-cursor");
    if (el) el.scrollIntoView({ block: "nearest" });
    const p = rows[state.cursor];
    announce(`${p.ward_name}. ${p.has_data ? p.lst_c + " degrees" : "no land pixels"}.`);
  }

  function wireWardKeys() {
    const onKey = (ev) => {
      const rows = filteredWards();
      if (ev.key === "ArrowDown") { ev.preventDefault(); moveCursor(state.cursor < 0 ? 1 : 1); }
      else if (ev.key === "ArrowUp") { ev.preventDefault(); moveCursor(-1); }
      else if (ev.key === "Home") { ev.preventDefault(); state.cursor = 0; renderWardList(); }
      else if (ev.key === "End") { ev.preventDefault(); state.cursor = rows.length - 1; renderWardList(); }
      else if (ev.key === "Enter" || ev.key === " ") {
        const p = rows[state.cursor];
        if (!p) return;
        ev.preventDefault();
        if (!p.has_data) return toast("That ward has no land pixels and cannot be analysed.");
        toggleWard(p.ward_id);
        renderWardList();
      }
    };
    $("wardList").addEventListener("keydown", onKey);
    $("wardSearch").addEventListener("keydown", (ev) => {
      if (["ArrowDown", "ArrowUp", "Enter"].includes(ev.key)) onKey(ev);
    });
    $("wardSearch").addEventListener("input", (e) => {
      state.filter = e.target.value; state.cursor = -1; renderWardList();
    });
  }

  /* ─────────────────── analysis ─────────────────── */
  function selectionPayload() {
    if (state.poly) return { kind: "polygon", coordinates: state.poly };
    if (state.bbox) return { kind: "bbox", bounds: state.bbox };
    if (state.selected.size) return { kind: "wards", ward_ids: [...state.selected] };
    return null;
  }

  function resetResults() {
    ["kDelta", "kArea", "kTemp", "kBudget"].forEach(id => { $(id).textContent = "—"; });
    ["kDeltaErr", "kAreaSub", "kBest"].forEach(id => { $(id).textContent = ""; });
    $("compare").hidden = true; $("breakdown").hidden = true;
    $("kpiCaveat").hidden = true;
    state.lastSummary = null;
    $("saveScenario").disabled = true;
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
      announce("Analysis failed: " + (err.message || err));
    } finally {
      if (myId === state.reqId) $("busyDot").hidden = true;
    }
  }

  /* ΔT is shown to ONE decimal, not two. The deployed checkpoint's held-out RMSE is ~2 K, so a
     second decimal is noise dressed as precision. The spread and the model error are reported
     next to it rather than folded into a single fake error bar — they measure different things. */
  const fmtDT = (v) => (v > 0 ? "+" : "") + v.toFixed(1);

  function renderResults(res) {
    const s = res.summary, none = s.no_change;
    state.lastSummary = { summary: s, items: res.items, selection: selectionPayload(),
                          interventions: body().interventions, season: state.season };
    $("saveScenario").disabled = false;

    $("kDelta").textContent = none ? "—" : fmtDT(s.mean_delta_t) + " °C";
    $("kDeltaErr").textContent = none ? ""
      : `spread ${fmtDT(s.delta_t_p10)} … ${fmtDT(s.delta_t_p90)}`
        + (s.model_rmse ? ` · model error ±${s.model_rmse} K` : "");

    $("kArea").textContent = s.treated_area_sqkm != null
      ? s.treated_area_sqkm.toFixed(2) + " km²" : "—";
    $("kAreaSub").textContent = s.area_sqkm
      ? `of ${s.area_sqkm.toFixed(2)} km² selected` : "";

    $("kTemp").textContent = none ? s.mean_t_base.toFixed(1) + " °C"
      : s.mean_t_base.toFixed(1) + " → " + (s.mean_t_base + s.mean_delta_t).toFixed(1);

    $("kBudget").textContent = (none || s.cooling_per_treated_km2 == null) ? "—"
      : fmtDT(s.cooling_per_treated_km2) + " K·km²/km²";
    $("kBest").textContent = none ? "set an intervention"
      : `best: ${s.best_name} (${fmtDT(s.best_delta_t)})`;

    const src = { measured: "measured per-ward roof share",
                  measured_city: "measured city-wide roof share",
                  user: "roof share set by you",
                  assumed: "roof share is an UNMEASURED assumption" }[s.roof_share_source];
    $("kpiCaveat").textContent =
      `Baseline is measured; ΔT is modelled and scaled by ${Math.round(s.roof_share * 100)}% `
      + `coverage (${src}). ${s.n_pixels.toLocaleString()} pixels analysed individually.`;
    $("kpiCaveat").hidden = false;

    const rows = res.items.filter(i => i.ward_id !== "area");
    if (rows.length > 1) {
      $("breakdownRows").innerHTML = rows.map(i => `
        <div class="bd-row"><span class="bd-name" title="${i.name}">${i.name}</span>
          <span class="bd-base">${i.t_base.toFixed(1)}°</span>
          <span class="bd-dt">${none ? "—" : fmtDT(i.delta_t)}</span></div>`).join("");
      $("breakdown").hidden = false;
    } else $("breakdown").hidden = true;

    if (!none) {
      announce(`Mean cooling ${fmtDT(s.mean_delta_t)} degrees across `
        + `${s.treated_area_sqkm} square kilometres treated.`);
    }
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
          <span class="cmp-val">${fmtDT(out[i])}</span>
        </div>`).join("");
      $("compare").hidden = false;
    } catch { /* comparison is a nicety; never block the main result */ }
  }

  /* ─────────────────── saved scenarios ─────────────────── */
  function loadScenarios() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch { return []; }          // private window, blocked storage — not a reason to break
  }

  function saveScenarios() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(state.scenarios)); }
    catch { toast("Could not save — this browser is blocking local storage.", true); }
  }

  function describeScenario(sc) {
    const iv = sc.interventions || {};
    const bits = [];
    if (iv.albedo_set) bits.push(`α→${(+iv.albedo_set).toFixed(2)}`);
    if (iv.ndvi_delta) bits.push(`+${(+iv.ndvi_delta).toFixed(2)} NDVI`);
    if (!bits.length) bits.push("baseline");
    bits.push(sc.season === "all" ? "all year" : sc.season);
    bits.push(`${Math.round((iv.roof_share ?? 0.5) * 100)}% cover`);
    return bits.join(" · ");
  }

  function saveCurrent() {
    const snap = state.lastSummary;
    if (!snap) return toast("Analyse a selection first.");
    const n = state.scenarios.length + 1;
    const label = snap.selection.kind === "wards"
      ? `${snap.selection.ward_ids.length} ward${snap.selection.ward_ids.length > 1 ? "s" : ""}`
      : snap.selection.kind === "polygon" ? "drawn zone" : "drawn area";
    const name = (prompt("Name this scenario:", `Scenario ${n} — ${label}`) || "").trim();
    if (!name) return;
    state.scenarios.unshift({
      id: "s" + Date.now().toString(36),
      name, ts: new Date().toISOString(),
      selection: snap.selection, interventions: snap.interventions, season: snap.season,
      summary: snap.summary, items: snap.items,
    });
    saveScenarios(); renderScenarios();
    toast(`Saved “${name}”.`);
    announce(`Scenario ${name} saved. ${state.scenarios.length} stored.`);
  }

  function renderScenarios() {
    const list = $("scenarioList");
    if (!state.scenarios.length) {
      list.innerHTML = `<p class="scen-empty">Nothing saved yet. Build a scenario, then
        <strong>Save current</strong> — it survives a page reload.</p>`;
      $("scenActions").hidden = true;
      $("scenarioDiff").hidden = true;
      return;
    }
    $("scenActions").hidden = false;
    list.innerHTML = state.scenarios.map(sc => {
      const picked = state.picked.includes(sc.id);
      const dt = sc.summary ? fmtDT(sc.summary.mean_delta_t) : "—";
      return `<div class="scen-row${picked ? " is-picked" : ""}">
        <button class="scen-pick" data-pick="${sc.id}" aria-pressed="${picked}"
                title="Pick for comparison">${picked ? (state.picked.indexOf(sc.id) === 0 ? "A" : "B") : ""}</button>
        <button class="scen-open" data-open="${sc.id}" title="Restore this scenario">
          ${escapeHtml(sc.name)}<span>${describeScenario(sc)}</span></button>
        <span class="scen-dt">${dt}</span>
        <button class="scen-del" data-del="${sc.id}" aria-label="Delete ${escapeHtml(sc.name)}">×</button>
      </div>`;
    }).join("");

    list.querySelectorAll("[data-pick]").forEach(b =>
      b.addEventListener("click", () => togglePick(b.dataset.pick)));
    list.querySelectorAll("[data-open]").forEach(b =>
      b.addEventListener("click", () => restoreScenario(b.dataset.open)));
    list.querySelectorAll("[data-del]").forEach(b =>
      b.addEventListener("click", () => deleteScenario(b.dataset.del)));
    renderDiff();
  }

  const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function togglePick(id) {
    const i = state.picked.indexOf(id);
    if (i >= 0) state.picked.splice(i, 1);
    else {
      state.picked.push(id);
      if (state.picked.length > MAX_PICK) state.picked.shift();
    }
    renderScenarios();
  }

  function deleteScenario(id) {
    const sc = state.scenarios.find(s => s.id === id);
    if (sc && !confirm(`Delete “${sc.name}”?`)) return;
    state.scenarios = state.scenarios.filter(s => s.id !== id);
    state.picked = state.picked.filter(p => p !== id);
    saveScenarios(); renderScenarios();
  }

  function restoreScenario(id) {
    const sc = state.scenarios.find(s => s.id === id);
    if (!sc) return;
    const iv = sc.interventions || {};
    state.iv.albedo = +iv.albedo_set || 0;
    state.iv.ndvi = +iv.ndvi_delta || 0;
    state.iv.roofShare = iv.roof_share ?? null;
    state.season = sc.season || "all";

    $("albedo").value = state.iv.albedo;
    $("ndvi").value = state.iv.ndvi;
    $("roofShare").value = state.iv.roofShare ?? state.roofShareDefault;
    renderRoofShare();
    document.querySelectorAll(".seg[data-season]").forEach(b => {
      const on = b.dataset.season === state.season;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-checked", String(on));
    });

    clearWardSelection(); clearDrawn();
    const sel = sc.selection || {};
    if (sel.kind === "wards") (sel.ward_ids || []).forEach(w => {
      if (state.wards.has(w)) { state.selected.add(w); setSel(w, true); }
    });
    else if (sel.kind === "bbox") state.bbox = sel.bounds;
    else if (sel.kind === "polygon") {
      state.poly = sel.coordinates;
      drawRing(state.poly, false);
    }

    if (state.map && sel.kind === "wards" && state.selected.size) fitSelection();
    syncSliders(); renderSelection(); renderWardList(); syncRankSelection(); analyze();
    toast(`Restored “${sc.name}”.`);
    announce(`Scenario ${sc.name} restored.`);
  }

  function fitSelection() {
    const b = new mapboxgl.LngLatBounds();
    let any = false;
    for (const f of (state.wardsGeo?.features || [])) {
      if (!state.selected.has(f.properties.ward_id)) continue;
      const walk = (c) => Array.isArray(c[0]) ? c.forEach(walk) : (b.extend(c), any = true);
      walk(f.geometry.coordinates);
    }
    if (any) state.map.fitBounds(b, { padding: 60, maxZoom: 14 });
  }

  /* A vs B. The metrics chosen are the ones a committee actually argues about: how much cooling,
     over how much area, for how much treated surface. */
  const DIFF_ROWS = [
    ["Mean ΔT", s => s.mean_delta_t, "°C", true],
    ["Selected area", s => s.area_sqkm, "km²", false],
    ["Surface treated", s => s.treated_area_sqkm, "km²", false],
    ["Cooling delivered", s => s.cooling_k_km2, "K·km²", true],
    ["Per km² treated", s => s.cooling_per_treated_km2, "K·km²/km²", true],
    ["Baseline temp", s => s.mean_t_base, "°C", false],
  ];

  function renderDiff() {
    const box = $("scenarioDiff");
    if (state.picked.length !== 2) { box.hidden = true; return; }
    const [a, b] = state.picked.map(id => state.scenarios.find(s => s.id === id));
    if (!a || !b || !a.summary || !b.summary) { box.hidden = true; return; }

    $("diffTitle").textContent = `A ${a.name}  ·  B ${b.name}`;
    const rows = DIFF_ROWS.map(([label, get, unit, lowerIsBetter]) => {
      const va = get(a.summary), vb = get(b.summary);
      if (va == null || vb == null) return "";
      const d = vb - va;
      const cls = Math.abs(d) < 1e-9 ? "" : (lowerIsBetter === (d < 0) ? "better" : "worse");
      return `<div class="diff-row">
        <span class="d-label">${label}</span>
        <span>${va.toFixed(2)}</span>
        <span>${vb.toFixed(2)}</span>
        <span class="d-delta ${cls}">${d > 0 ? "+" : ""}${d.toFixed(2)}</span>
      </div>`;
    }).join("");
    $("diffRows").innerHTML =
      `<div class="diff-row diff-head-row"><span>metric</span><span>A</span><span>B</span>
        <span>B−A</span></div>` + rows;
    box.hidden = false;
  }

  function exportScenarios() {
    if (!state.scenarios.length) return toast("Nothing saved to export.");
    const v = (state.meta.vintage || {});
    const head = [
      "# UrbanHeat Planner — saved scenarios",
      `# exported,${new Date().toISOString()}`,
      `# model,${state.meta.model}`,
      `# model_rmse_K,${state.meta.model_rmse ?? ""}`,
      `# baseline,${v.sensor || "Landsat 8/9"} ${v.first_date || v.start || ""} to ${v.last_date || v.end || ""}`,
      `# baseline_scenes,${v.n_scenes ?? ""}`,
      "# NOTE baseline LST is measured; delta_t is modelled and coverage-scaled",
      "# NOTE delta_t_p10/p90 are the spread across pixels, not a model error bar",
    ].join("\n");
    const cols = ["name", "saved_at", "season", "albedo_set", "ndvi_delta", "roof_share",
                  "roof_share_source", "selection", "n_units", "area_sqkm", "treated_area_sqkm",
                  "mean_t_base_c", "mean_delta_t_c", "delta_t_p10", "delta_t_p90",
                  "cooling_k_km2", "cooling_per_treated_km2"];
    const lines = state.scenarios.map(sc => {
      const s = sc.summary || {}, iv = sc.interventions || {};
      const sel = sc.selection || {};
      const which = sel.kind === "wards" ? (sel.ward_ids || []).join(" ") : sel.kind;
      return [sc.name, sc.ts, s.season, iv.albedo_set || "", iv.ndvi_delta || "",
              s.roof_share, s.roof_share_source, which, s.n, s.area_sqkm, s.treated_area_sqkm,
              s.mean_t_base, s.mean_delta_t, s.delta_t_p10, s.delta_t_p90,
              s.cooling_k_km2, s.cooling_per_treated_km2]
        .map(x => `"${String(x ?? "").replace(/"/g, '""')}"`).join(",");
    });
    download(new Blob([head + "\n" + cols.join(",") + "\n" + lines.join("\n") + "\n"],
                      { type: "text/csv" }), "urbanheat-scenarios.csv");
    toast(`Exported ${state.scenarios.length} scenario${state.scenarios.length > 1 ? "s" : ""}.`);
  }

  function download(blob, filename) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  /* ─────────────── onboarding ─────────────── */
  function maybeCoach() {
    let seen = false;
    try { seen = localStorage.getItem(COACH_KEY) === "1"; } catch { seen = false; }
    if (!seen) $("coach").hidden = false;
  }

  function dismissCoach() {
    $("coach").hidden = true;
    try { localStorage.setItem(COACH_KEY, "1"); } catch { /* nothing to do */ }
  }

  /* ─────────────────── UI wiring ─────────────────── */
  function syncSliders() {
    $("albedoOut").textContent = state.iv.albedo ? "α → " + state.iv.albedo.toFixed(2) : "off";
    $("ndviOut").textContent = state.iv.ndvi ? "+" + state.iv.ndvi.toFixed(2) + " NDVI" : "off";
    document.querySelectorAll(".preset").forEach(p => p.classList.toggle("is-active",
      Math.abs(+p.dataset.albedo - state.iv.albedo) < 1e-9
      && Math.abs(+p.dataset.ndvi - state.iv.ndvi) < 1e-9));
  }

  /* The slider is an OVERRIDE, not the live value. Until it is touched the backend uses the
     measured per-ward roof share, which differs ward to ward and cannot be shown as one number;
     the slider sits at the city-wide figure purely as a starting point. */
  function renderRoofShare() {
    const override = state.iv.roofShare !== null;
    const measured = !!(state.meta && state.meta.roof_share_measured);
    $("roofShareOut").textContent = override
      ? Math.round(state.iv.roofShare * 100) + "%"
      : (measured ? "measured" : Math.round(state.roofShareDefault * 100) + "%");
    $("roofShareReset").hidden = !(override && measured);
    $("roofShareNote").textContent = override
      ? (measured
        ? "Overriding the measured per-ward roof share with one flat value."
        : "What share of built surface is actually treated. This scales every result.")
      : (measured
        ? "Measured per ward from building footprints. Drag to override with a flat assumption."
        : "What share of built surface is actually treated. This scales every result — it is an "
          + "assumption, not a measurement.");
  }

  function wireUI() {
    document.querySelectorAll(".tool[data-tool]").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll(".tool[data-tool]").forEach(x => x.classList.remove("is-active"));
      b.classList.add("is-active");
      cancelTracing();
      state.tool = b.dataset.tool;
      const box = state.tool === "draw", poly = state.tool === "polygon";
      $("map").classList.toggle("is-drawing", box);
      $("map").classList.toggle("is-tracing", poly);
      $("mapHint").hidden = poly;
      $("mapHint").textContent = box
        ? "Drag on the map to sample a custom area"
        : "Click wards to build a scenario · drag to pan · scroll to zoom";
      traceBar(0);
      // Only the rubber-band box takes over dragging. Polygon tracing keeps pan available for
      // repositioning, and tells a tap from a drag by distance instead.
      if (state.map) box ? state.map.dragPan.disable() : state.map.dragPan.enable();
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

    $("choropleth").addEventListener("change", e => {
      state.choropleth = e.target.value;
      applyChoropleth();
      if (state.choropleth === "delta" && !state.rank.length) loadRank();
    });

    $("albedo").addEventListener("input", e => { state.iv.albedo = +e.target.value; syncSliders(); analyze(); });
    $("ndvi").addEventListener("input", e => { state.iv.ndvi = +e.target.value; syncSliders(); analyze(); });
    document.querySelectorAll(".preset").forEach(p => p.addEventListener("click", () => {
      state.iv.albedo = +p.dataset.albedo; state.iv.ndvi = +p.dataset.ndvi;
      $("albedo").value = state.iv.albedo; $("ndvi").value = state.iv.ndvi;
      syncSliders(); analyze();
    }));
    syncSliders();

    document.querySelectorAll(".seg[data-season]").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll(".seg[data-season]").forEach(x => {
        x.classList.remove("is-active"); x.setAttribute("aria-checked", "false");
      });
      b.classList.add("is-active"); b.setAttribute("aria-checked", "true");
      state.season = b.dataset.season;
      analyze(); if (state.rank.length) loadRank();
    }));

    document.querySelectorAll(".seg[data-rank]").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll(".seg[data-rank]").forEach(x => {
        x.classList.remove("is-active"); x.setAttribute("aria-checked", "false");
      });
      b.classList.add("is-active"); b.setAttribute("aria-checked", "true");
      state.rankBy = b.dataset.rank;
      renderRank();
    }));

    $("roofShare").addEventListener("input", e => {
      state.iv.roofShare = +e.target.value;       // now an explicit override
      renderRoofShare(); analyze();
    });
    $("roofShareReset").addEventListener("click", () => {
      state.iv.roofShare = null;                  // back to the measured per-ward values
      $("roofShare").value = state.roofShareDefault;
      renderRoofShare(); analyze();
    });

    $("rankRefresh").addEventListener("click", loadRank);
    $("exportBtn").addEventListener("click", exportCsv);
    $("clearSel").addEventListener("click", clearSelection);
    $("aboutBtn").addEventListener("click", () => $("aboutDlg").showModal());
    $("saveScenario").addEventListener("click", saveCurrent);
    $("saveScenario").disabled = true;
    $("exportScenarios").addEventListener("click", exportScenarios);
    $("printSheet").addEventListener("click", () => window.print());
    $("closeDiff").addEventListener("click", () => { state.picked = []; renderScenarios(); });
    $("coachDone").addEventListener("click", dismissCoach);
    $("helpBtn").addEventListener("click", () => { $("coach").hidden = false; });

    wireWardKeys();

    document.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        if (!$("coach").hidden) return dismissCoach();
        if (state.tracing.length) return cancelTracing();
      }
      if (ev.key === "Enter" && state.tool === "polygon" && state.tracing.length >= 3) {
        finishPolygon();
      }
    });
  }

  /* ─────────────────── prioritisation ─────────────────── */
  const RANK_SORT = {
    delta: (a, b) => a.delta_t - b.delta_t,
    t_base: (a, b) => b.t_base - a.t_base,
    efficiency: (a, b) => (a.cooling_per_treated_km2 ?? 0) - (b.cooling_per_treated_km2 ?? 0),
  };

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
      state.rank = res.items;
      renderRank();
      pushRankToMap();
      announce(`${state.rank.length} wards ranked.`);
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
    const sorted = [...state.rank].sort(RANK_SORT[state.rankBy]);
    const metric = {
      delta: (w) => fmtDT(w.delta_t),
      t_base: (w) => w.t_base.toFixed(1) + "°",
      efficiency: (w) => w.cooling_per_treated_km2 == null ? "—"
        : fmtDT(w.cooling_per_treated_km2),
    }[state.rankBy];

    $("rankList").innerHTML = sorted.map((w, i) => `
      <button class="rank-row${state.selected.has(w.ward_id) ? " is-selected" : ""}"
              role="listitem" data-id="${w.ward_id}"
              title="${escapeHtml(w.name)} — baseline ${w.t_base}°C, ${Math.round(w.built_frac * 100)}% built-up, ${w.treated_sqkm} km² treatable">
        <span class="rank-n">${i + 1}</span>
        <span class="rank-name">${escapeHtml(w.name)}</span>
        <span class="rank-base">${w.t_base.toFixed(1)}°</span>
        <span class="rank-dt">${metric(w)}</span>
      </button>`).join("");
    $("rankList").querySelectorAll(".rank-row").forEach(b =>
      b.addEventListener("click", () => toggleWard(b.dataset.id)));
    $("rankNote").textContent = {
      delta: "Ranked by predicted cooling for the current scenario — click to add to the selection.",
      t_base: "Ranked by today's measured temperature — where the heat problem is worst.",
      efficiency: "Ranked by cooling delivered per km² of surface treated — the budget view.",
    }[state.rankBy];
  }

  async function exportCsv() {
    if (!selectionPayload()) return toast("Select an area first.");
    try {
      const r = await fetch("/api/export", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()),
      });
      if (!r.ok) throw new Error(await r.text());
      download(await r.blob(), `urbanheat-scenario-${state.season}.csv`);
      toast("Scenario exported.");
    } catch (err) { toast(String(err.message || err), true); }
  }

  boot();
})();
