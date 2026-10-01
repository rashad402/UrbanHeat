/* UrbanHeat Planner — front end.
 *
 * The map is a server-rendered Earth Engine scene (one <img> per layer) with an SVG interaction
 * layer on top. No tile/WebGL map library: the scene is embedded as data URIs, so there are no
 * ongoing network requests and the view always paints. Zoom/pan are CSS transforms on the scene
 * wrapper, so the imagery and the SVG stay registered automatically.
 *
 * Geography <-> screen: the scene image covers exactly [west,south,east,north] in a plate-carree
 * layout, so lon/lat map linearly onto the SVG user-space box.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const api = (p, opts) => fetch(p, opts).then(r => {
    if (!r.ok) return r.json().then(e => { throw new Error(e.detail || r.statusText); });
    return r.json();
  });

  const state = {
    bounds: null, imgW: 0, imgH: 0,
    wards: new Map(),            // ward_id -> {props, el}
    selected: new Set(),
    tool: "select",
    bbox: null,                  // drawn area [w,s,e,n]
    view: { scale: 1, x: 0, y: 0 },
    iv: { albedo: 0, ndvi: 0 },
    reqId: 0,
  };

  /* ───────────────────────── boot ───────────────────────── */
  async function boot() {
    try {
      const [scene, wards, meta] = await Promise.all([
        api("/api/scene"), api("/api/wards"), api("/api/meta"),
      ]);
      state.bounds = scene.bounds;
      const lam = (meta.model.match(/lambda([0-9.]+?)\.pt/) || [])[1];
      $("chipModel").textContent = lam ? `PINN · λ=${lam}` : meta.model;
      $("chipWards").textContent = meta.n_wards + " wards with data";

      $("legendBar").style.background = `linear-gradient(90deg, ${scene.palette.join(",")})`;
      $("legLo").textContent = scene.lst_range[0] + "°C";
      $("legHi").textContent = scene.lst_range[1] + "°C";

      await loadScene(scene);
      drawWards(wards);
      autoFit();
      $("stageLoading").hidden = true;
      wireUI();
    } catch (err) {
      $("stageLoading").innerHTML =
        `<p style="color:#f06a5d;max-width:34ch">Could not reach the backend.<br><br>
         <span style="color:#8a8498">Start it with<br>
         <code style="font-family:IBM Plex Mono,monospace">uvicorn api.planner:app --port 8080</code></span></p>`;
      console.error(err);
    }
  }

  function loadScene(scene) {
    return new Promise((resolve) => {
      const sat = $("satImg"), lst = $("lstImg");
      let n = 0;
      const done = () => {
        if (++n < 2) return;
        state.imgW = sat.naturalWidth || 706;
        state.imgH = sat.naturalHeight || 1100;
        const el = $("scene");
        el.style.width = state.imgW + "px";
        el.style.height = state.imgH + "px";
        $("overlay").setAttribute("viewBox", `0 0 ${state.imgW} ${state.imgH}`);
        resolve();
      };
      sat.onload = done; lst.onload = done;
      sat.onerror = done; lst.onerror = done;
      sat.src = scene.sat; lst.src = scene.lst;
    });
  }

  /* ─────────────────── projection + ward layer ─────────────────── */
  const toX = (lon) => (lon - state.bounds[0]) / (state.bounds[2] - state.bounds[0]) * state.imgW;
  const toY = (lat) => (state.bounds[3] - lat) / (state.bounds[3] - state.bounds[1]) * state.imgH;

  function ringPath(ring) {
    let d = "";
    for (let i = 0; i < ring.length; i++) {
      d += (i ? "L" : "M") + toX(ring[i][0]).toFixed(1) + " " + toY(ring[i][1]).toFixed(1);
    }
    return d + "Z";
  }

  function geomPath(g) {
    const polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
    return polys.map(p => p.map(ringPath).join("")).join("");
  }

  function drawWards(fc) {
    const layer = $("wardLayer");
    const frag = document.createDocumentFragment();
    for (const f of fc.features) {
      const p = f.properties;
      const el = document.createElementNS("http://www.w3.org/2000/svg", "path");
      el.setAttribute("d", geomPath(f.geometry));
      el.setAttribute("class", "ward" + (p.has_data ? "" : " no-data"));
      el.dataset.id = p.ward_id;
      const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
      t.textContent = p.has_data
        ? `${p.ward_name} — ${p.lst_c}°C, ${Math.round(p.built_frac * 100)}% built-up`
        : `${p.ward_name} — no land pixels (water-dominated)`;
      el.appendChild(t);
      if (p.has_data) el.addEventListener("click", (e) => { e.stopPropagation(); toggleWard(p.ward_id); });
      state.wards.set(p.ward_id, { props: p, el });
      frag.appendChild(el);
    }
    layer.appendChild(frag);
  }

  /* ─────────────────── view transform ─────────────────── */
  function applyView() {
    const { scale, x, y } = state.view;
    $("scene").style.transform =
      `translate(-50%,-50%) translate(${x}px, ${y}px) scale(${scale})`;
  }

  function fitScene() {
    const vp = $("viewport").getBoundingClientRect();
    // Guard: the pane can report zero size before layout settles, which would yield a
    // negative scale and flip the whole scene.
    if (vp.width < 40 || vp.height < 40) return false;
    const pad = 28;
    state.view.scale = Math.max(0.05,
      Math.min((vp.width - pad) / state.imgW, (vp.height - pad) / state.imgH));
    state.view.x = state.view.y = 0;
    state.fitted = true;
    applyView();
    return true;
  }

  /* Fit as soon as the viewport actually has a size (and only once, so a later resize
     never throws away the planner's pan/zoom). */
  function autoFit() {
    if (fitScene()) return;
    const ro = new ResizeObserver(() => { if (fitScene()) ro.disconnect(); });
    ro.observe($("viewport"));
  }

  function zoomBy(factor, cx, cy) {
    const vp = $("viewport").getBoundingClientRect();
    const px = (cx ?? vp.width / 2) - vp.width / 2 - state.view.x;
    const py = (cy ?? vp.height / 2) - vp.height / 2 - state.view.y;
    const next = Math.min(12, Math.max(0.15, state.view.scale * factor));
    const k = next / state.view.scale;
    state.view.x -= px * (k - 1);
    state.view.y -= py * (k - 1);
    state.view.scale = next;
    applyView();
  }

  /* screen point -> lon/lat */
  function toLngLat(clientX, clientY) {
    const r = $("satImg").getBoundingClientRect();
    const fx = (clientX - r.left) / r.width, fy = (clientY - r.top) / r.height;
    return [
      state.bounds[0] + fx * (state.bounds[2] - state.bounds[0]),
      state.bounds[3] - fy * (state.bounds[3] - state.bounds[1]),
    ];
  }

  /* ─────────────────── selection ─────────────────── */
  function toggleWard(id) {
    state.bbox = null;
    $("drawRect").hidden = true;
    state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
    syncWardStyles(); renderSelection(); analyze();
  }

  function syncWardStyles() {
    for (const [id, w] of state.wards) {
      w.el.classList.toggle("is-selected", state.selected.has(id));
    }
  }

  function clearSelection() {
    state.selected.clear(); state.bbox = null;
    $("drawRect").hidden = true;
    syncWardStyles(); renderSelection(); resetResults();
  }

  function renderSelection() {
    const box = $("selectionBox");
    const n = state.selected.size;
    $("clearSel").hidden = !(n || state.bbox);

    if (state.bbox) {
      box.innerHTML = `<div class="sel-summary"><span class="sel-count">1</span>
        <span class="sel-unit">custom area</span></div>
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
      const p = state.wards.get(id).props;
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
  function analyze() {
    clearTimeout(timer);
    timer = setTimeout(runAnalyze, 160);
  }

  async function runAnalyze() {
    const sel = selectionPayload();
    if (!sel) return resetResults();
    const myId = ++state.reqId;
    $("busyDot").hidden = false;
    try {
      const body = { selection: sel, interventions: { albedo_set: state.iv.albedo, ndvi_delta: state.iv.ndvi } };
      const res = await api("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      if (myId !== state.reqId) return;                 // a newer request superseded this one
      renderResults(res);
      if (sel.kind === "wards") runCompare(sel, myId); else $("compare").hidden = true;
    } catch (err) {
      if (myId !== state.reqId) return;
      resetResults();
      $("kBest").textContent = String(err.message || err);
    } finally {
      if (myId === state.reqId) $("busyDot").hidden = true;
    }
  }

  function renderResults(res) {
    const s = res.summary;
    const none = s.no_change;
    $("kDelta").textContent = none ? "—" : s.mean_delta_t.toFixed(2) + " °C";
    $("kArea").textContent = s.area_sqkm ? s.area_sqkm.toFixed(2) + " km²" : "—";
    $("kTemp").textContent = none
      ? s.mean_t_base.toFixed(1) + " °C"
      : s.mean_t_base.toFixed(1) + " → " + (s.mean_t_base + s.mean_delta_t).toFixed(1);
    $("kBest").textContent = none ? "set an intervention" : `${s.best_name} (${s.best_delta_t.toFixed(2)} °C)`;

    const rows = res.items.filter(i => i.ward_id !== "area");
    if (rows.length > 1) {
      $("breakdownRows").innerHTML = rows.map(i => `
        <div class="bd-row"><span class="bd-name" title="${i.name}">${i.name}</span>
          <span class="bd-base">${i.t_base.toFixed(1)}°</span>
          <span class="bd-dt">${none ? "—" : i.delta_t.toFixed(2)}</span></div>`).join("");
      $("breakdown").hidden = false;
    } else {
      $("breakdown").hidden = true;
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
          body: JSON.stringify({ selection: sel, interventions: { albedo_set: st.albedo_set, ndvi_delta: st.ndvi_delta } }),
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
    // tools
    document.querySelectorAll(".tool[data-tool]").forEach(b => b.addEventListener("click", () => {
      document.querySelectorAll(".tool[data-tool]").forEach(x => x.classList.remove("is-active"));
      b.classList.add("is-active");
      state.tool = b.dataset.tool;
      $("viewport").classList.toggle("is-drawing", state.tool === "draw");
      $("mapHint").textContent = state.tool === "draw"
        ? "Drag on the map to sample a custom area"
        : "Click wards to build a scenario · drag to pan · scroll to zoom";
    }));
    $("zoomIn").addEventListener("click", () => zoomBy(1.3));
    $("zoomOut").addEventListener("click", () => zoomBy(1 / 1.3));
    $("zoomReset").addEventListener("click", fitScene);

    // overlay opacity
    $("opacity").addEventListener("input", e => { $("lstImg").style.opacity = e.target.value; });

    // interventions
    const sync = () => {
      $("albedoOut").textContent = state.iv.albedo ? "α → " + state.iv.albedo.toFixed(2) : "off";
      $("ndviOut").textContent = state.iv.ndvi ? "+" + state.iv.ndvi.toFixed(2) + " NDVI" : "off";
      document.querySelectorAll(".preset").forEach(p => p.classList.toggle("is-active",
        Math.abs(+p.dataset.albedo - state.iv.albedo) < 1e-9 && Math.abs(+p.dataset.ndvi - state.iv.ndvi) < 1e-9));
    };
    $("albedo").addEventListener("input", e => { state.iv.albedo = +e.target.value; sync(); analyze(); });
    $("ndvi").addEventListener("input", e => { state.iv.ndvi = +e.target.value; sync(); analyze(); });
    document.querySelectorAll(".preset").forEach(p => p.addEventListener("click", () => {
      state.iv.albedo = +p.dataset.albedo; state.iv.ndvi = +p.dataset.ndvi;
      $("albedo").value = state.iv.albedo; $("ndvi").value = state.iv.ndvi;
      sync(); analyze();
    }));
    sync();

    $("clearSel").addEventListener("click", clearSelection);
    $("aboutBtn").addEventListener("click", () => $("aboutDlg").showModal());

    wirePointer();
  }

  function wirePointer() {
    const vp = $("viewport");
    let mode = null, sx = 0, sy = 0, ox = 0, oy = 0, start = null, moved = false;

    vp.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = vp.getBoundingClientRect();
      zoomBy(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });

    vp.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      vp.setPointerCapture(e.pointerId);
      moved = false;
      if (state.tool === "draw") {
        mode = "draw";
        start = toLngLat(e.clientX, e.clientY);
        const r = $("drawRect");
        r.hidden = false;
        r.setAttribute("x", toX(start[0])); r.setAttribute("y", toY(start[1]));
        r.setAttribute("width", 0); r.setAttribute("height", 0);
      } else {
        mode = "pan"; sx = e.clientX; sy = e.clientY; ox = state.view.x; oy = state.view.y;
        vp.classList.add("is-panning");
      }
    });

    vp.addEventListener("pointermove", (e) => {
      if (!mode) return;
      if (Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) > 3) moved = true;
      if (mode === "pan") {
        state.view.x = ox + (e.clientX - sx);
        state.view.y = oy + (e.clientY - sy);
        applyView();
      } else {
        const cur = toLngLat(e.clientX, e.clientY);
        const x1 = toX(Math.min(start[0], cur[0])), x2 = toX(Math.max(start[0], cur[0]));
        const y1 = toY(Math.max(start[1], cur[1])), y2 = toY(Math.min(start[1], cur[1]));
        const r = $("drawRect");
        r.setAttribute("x", x1); r.setAttribute("y", y1);
        r.setAttribute("width", Math.max(0, x2 - x1)); r.setAttribute("height", Math.max(0, y2 - y1));
      }
    });

    vp.addEventListener("pointerup", (e) => {
      const was = mode; mode = null;
      vp.classList.remove("is-panning");
      if (was !== "draw") return;
      const cur = toLngLat(e.clientX, e.clientY);
      const w = Math.min(start[0], cur[0]), ee = Math.max(start[0], cur[0]);
      const s = Math.min(start[1], cur[1]), n = Math.max(start[1], cur[1]);
      if (!moved || (ee - w) < 1e-4 || (n - s) < 1e-4) { $("drawRect").hidden = true; return; }
      state.selected.clear(); syncWardStyles();
      state.bbox = [w, s, ee, n];
      renderSelection(); analyze();
    });

    vp.addEventListener("pointercancel", () => { mode = null; vp.classList.remove("is-panning"); });
  }

  boot();
})();
