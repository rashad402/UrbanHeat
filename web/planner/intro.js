/* UrbanHeat Planner — opening scene controller.

   Three jobs: draw the starfield, hold the scene until the app says it is ready, and get out of
   the way. The markup is in index.html (so it paints before any script runs and the app never
   flashes underneath) and the look is in intro.css.

   THE HANDSHAKE
   app.js calls UrbanIntro.ready() once the map has SETTLED: its layers are in and every tile it
   asked for has arrived (Mapbox's "idle"), so the scene lifts onto a map with something on it,
   not onto an empty frame that fills in afterwards. It also calls ready() on every path where the
   map never will — no token, no WebGL, backend down — because there the planner needs to read the
   failure message, not wait behind a splash.
   The scene never leaves before MIN_MS, so the wordmark and caption always finish, and never
   sooner than HOLD_MS after the loader fills, so completion registers. MAX_MS is a backstop: if
   nothing ever calls ready(), it still leaves.

   WHY THE EXIT IS WEB ANIMATIONS, NOT A CSS CLASS
   It used to flip a class and let four independent transitions run with different durations and
   easings, which never agreed with each other. One timeline, one easing family, run on the
   compositor (transform and opacity only), reads as a single camera move. Each animation starts
   from the element's CURRENT state, so skipping mid-entrance does not jump.

   WHY TIMERS DECIDE WHEN IT LEAVES
   requestAnimationFrame is throttled to nothing in a background tab, and a splash must never be
   able to trap the app.

   While it is up the app behind it is inert, so Tab cannot walk focus into controls nobody can
   see. Any click or key skips it. */
(() => {
  "use strict";

  const el = document.getElementById("intro");
  if (!el || document.documentElement.classList.contains("no-intro")) return;

  const SEEN_KEY = "urbanheat.intro.v1";
  const MIN_MS = 4600;                 // the entrance is fully composed at about 3.4s
  const HOLD_MS = 800;                 // after the loader fills, before the scene starts to leave
  const MAX_MS = 18000;
  const EXIT_MS = 1700;                // one timeline for the whole exit
  const EASE = "cubic-bezier(.65,0,.25,1)";

  const t0 = performance.now();
  const regions = document.querySelectorAll(".topbar, .shell, .statusbar");
  regions.forEach((r) => r.setAttribute("inert", ""));

  let leaving = false, stopStars = () => {};

  function exit() {
    const q = (s) => el.querySelector(s);
    const timing = (duration, easing) => ({ duration, easing, fill: "forwards" });

    if (!el.animate) {                                   // very old browser: a plain fade will do
      el.style.transition = "opacity .9s ease"; el.style.opacity = "0";
      return;
    }

    // The planet descends: the camera drops toward the surface.
    q(".intro-planet").animate({ transform: "translateY(-7%) scale(1.55)" }, timing(EXIT_MS, EASE));
    // The sky dims as it falls away.
    q(".intro-sky").animate({ opacity: 0 }, timing(EXIT_MS * 0.9, "ease-in"));
    // The wordmark leaves first, so the move starts on the text and not on the whole frame.
    q(".intro-center").animate(
      { opacity: 0, transform: "translateY(-50%) scale(1.07)", filter: "blur(8px)" },
      timing(EXIT_MS * 0.42, "cubic-bezier(.4,0,1,1)"));
    q(".intro-hint").animate({ opacity: 0 }, timing(280, "ease-out"));
    // The overlay itself holds fully opaque for the first part of the move, then dissolves — the
    // app is only revealed once the camera is already moving.
    el.animate(
      [{ opacity: 1, offset: 0 },
       { opacity: 1, offset: 0.36, easing: "cubic-bezier(.4,0,.2,1)" },
       { opacity: 0, offset: 1 }],
      { duration: EXIT_MS, easing: "linear", fill: "forwards" });
  }

  function leave() {
    if (leaving) return;
    leaving = true;
    stopStars();                          // free the main thread for the exit; the last frame stays
    el.classList.add("is-leaving");
    regions.forEach((r) => r.removeAttribute("inert"));
    document.removeEventListener("keydown", leave);
    el.removeEventListener("pointerdown", leave);
    exit();
    setTimeout(() => {
      el.hidden = true;
      el.style.display = "none";
      try { sessionStorage.setItem(SEEN_KEY, "1"); } catch { /* private window: it just plays again */ }
    }, EXIT_MS + 120);
  }

  window.UrbanIntro = {
    ready() {
      if (leaving) return;
      el.classList.add("is-ready");                       // the loader comet fills its track
      const wait = Math.max(MIN_MS - (performance.now() - t0), HOLD_MS);
      setTimeout(leave, wait);
    },
    skip: leave,
  };

  document.addEventListener("keydown", leave);
  el.addEventListener("pointerdown", leave);
  setTimeout(leave, MAX_MS);

  /* ───────────── starfield ─────────────
     Stars are drawn once immediately, so the scene is never empty, and then twinkle at ~30fps —
     a twinkle does not need 60, and the main thread is busy starting the map. A third of them sit
     along the same diagonal band the CSS Milky Way glows on, so the band reads as dense with stars
     and not just as a gradient. A few are bright enough to get a soft cross. */
  const canvas = document.getElementById("introStars");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  let stars = [], W = 0, H = 0, dpr = 1, raf = 0, running = true, lastDraw = 0;

  const rnd = (a, b) => a + Math.random() * (b - a);
  const gauss = () => (Math.random() + Math.random() + Math.random() + Math.random() - 2) / 2;

  function build() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const n = Math.max(420, Math.min(1500, Math.round((W * H) / 1000)));
    stars = [];
    // the band runs along the direction of the CSS gradient: 115deg, through the left-centre
    const ang = (115 - 90) * Math.PI / 180, cx = W * 0.40, cy = H * 0.50;
    const dx = Math.cos(ang), dy = Math.sin(ang), nx = -dy, ny = dx;
    for (let i = 0; i < n; i++) {
      let x, y, band = i % 5 < 2;
      if (band) {
        const along = rnd(-1.1, 1.1) * Math.hypot(W, H) * 0.5, across = gauss() * Math.min(W, H) * 0.13;
        x = cx + dx * along + nx * across; y = cy + dy * along + ny * across;
      } else { x = rnd(0, W); y = rnd(0, H); }
      const bright = Math.random() < 0.018;
      stars.push({
        x, y,
        r: bright ? rnd(1.2, 1.9) : rnd(0.3, 1.2),
        a: bright ? rnd(.9, 1) : rnd(.3, .95) * (band ? .85 : 1),
        s: rnd(.4, 1.6), p: rnd(0, 6.28), bright,
        // a faint warm or cool cast, as in a long exposure
        c: Math.random() < .18 ? "255,226,196" : Math.random() < .3 ? "196,214,255" : "255,255,255",
      });
    }
  }

  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    for (const s of stars) {
      const tw = .72 + .28 * Math.sin(t * .001 * s.s + s.p);
      const a = s.a * tw;
      ctx.fillStyle = `rgba(${s.c},${a.toFixed(3)})`;
      ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, 6.2832); ctx.fill();
      if (s.bright) {
        ctx.strokeStyle = `rgba(${s.c},${(a * .45).toFixed(3)})`; ctx.lineWidth = .6;
        const L = s.r * 5.5;
        ctx.beginPath(); ctx.moveTo(s.x - L, s.y); ctx.lineTo(s.x + L, s.y);
        ctx.moveTo(s.x, s.y - L); ctx.lineTo(s.x, s.y + L); ctx.stroke();
      }
    }
  }

  function loop(t) {
    if (!running) return;
    if (t - lastDraw >= 32) { lastDraw = t; draw(t); }
    raf = requestAnimationFrame(loop);
  }

  build();
  draw(0);                                // painted before the first animation frame
  raf = requestAnimationFrame(loop);
  const onResize = () => { if (running) { build(); draw(performance.now()); } };
  window.addEventListener("resize", onResize);
  stopStars = () => {
    running = false; cancelAnimationFrame(raf);
    window.removeEventListener("resize", onResize);
  };
})();
