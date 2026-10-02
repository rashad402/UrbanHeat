"""UrbanHeat Planner — decision-support backend for municipal planners (plan §11, §12).

Run:
    uvicorn api.planner:app --port 8080
    # then open http://localhost:8080

Serves the planning console in web/planner/ plus:

    GET  /api/scene     Sentinel-2 true colour + Landsat LST, rendered server-side by Earth
                        Engine and returned as data URIs (no external tiles -> always renders)
    GET  /api/wards     ward polygons (simplified) with measured baseline statistics
    POST /api/analyze   run the trained PINN over a selection and return the predicted LST

Selections are either a set of wards (instant — uses the per-ward feature means the model was
trained on) or a drawn rectangle (samples Earth Engine live, ~1-2 s).

Baselines are always the MEASURED Landsat LST; the PINN supplies the response (delta T). See
api/inference.py for why the deployed checkpoint is lambda=0.5 rather than the highest-R2 one.
"""

import base64
import json
import math
import os
import sys
import threading
import time
import urllib.request

import ee
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.inference import PinnModel, whatif, built_fraction   # noqa: E402
from data_engine.build_dataset import load_config, aoi_geometry, build_collection  # noqa: E402
from models.features import INPUT_COLUMNS                      # noqa: E402

CFG_PATH = "configs/data_config.yaml"
WARDS_PATH = "configs/wards_kochi.geojson"
PARQUET = "data/processed/kochi_samples.parquet"
WEB_DIR = "web/planner"
SIMPLIFY_TOL = 0.00025          # ~28 m — keeps the SVG light without visible distortion

app = FastAPI(title="UrbanHeat Planner")
S = {}


def _load_env(path=".env"):
    """Minimal .env reader — keeps the Mapbox token out of committed source."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")

PALETTE = ["313695", "4575b4", "74add1", "abd9e9", "fee090",
           "fdae61", "f46d43", "d73027", "a50026"]
# Colour range of the thermal layer and its legend. This is only the FALLBACK: at startup the range
# is derived from the data (see _stretch_lst_range). A fixed 28-46 degC span is far wider than the
# city's real spread, so nearly every land pixel landed in the same orange band and wards were
# indistinguishable on the map.
LST_MIN, LST_MAX = 31, 41

# Why a ward can be missing from the model's training table. This is stated to the planner, so it
# has to be true of the data actually built: ERA5-Land is an 11 km grid with a coarse land-sea mask,
# and the coastal cells covering southern and western Kochi are masked as sea. Their climate bands
# are null, and the dataset builder drops any pixel with a null band (dropNulls), so ~20% of the
# corporation never reached the table even though Landsat covers it fully.
NO_DATA_REASON = (
    "Outside current model coverage. The climate inputs (ERA5-Land, 11 km) have no land cell "
    "over this coastal area, so the model has nothing to run on here. This is a data gap, not "
    "a judgement that the ward has no land."
)


# Share of BUILT-UP area that is actually roof. Measured per ward by
# scripts/build_roof_share.py from Open Buildings footprints; this constant is only the fallback
# for when that file has not been generated. It multiplies every headline number, so the UI
# exposes it as an adjustable assumption either way.
ROOF_SHARE_DEFAULT = 0.5
ROOF_SHARE_PATH = "configs/roof_share_kochi.json"

# Earth Engine signs tile URLs and they expire. A dead URL fails silently — the thermal layer
# just stops drawing, with no error anywhere — so the URL is re-minted on a timer and on demand
# via /api/refresh_tiles.
TILE_TTL_S = 45 * 60

# Pixels sampled for a drawn selection. The area is measured from the geometry, never from
# this count — see _region_area_sqkm.
DRAW_SAMPLE_PIXELS = 600
SCENE_MARGIN = 0.12      # fractional padding around the corporation, for geographic context
MIN_ASPECT = 1.0         # widen east-west so the map fills a landscape viewport
TILE_DIM = 1100          # Earth Engine refuses a single thumbnail much above this (HTTP 400)
SAT_GRID = 2             # 2x2 tiles -> ~2200 px of satellite detail (~10 m/px, S2 native)
LST_DIM = 1100           # Landsat is 30 m, so one tile already over-samples it


def _fetch_b64(url, mime="image/png"):
    data = urllib.request.urlopen(url, timeout=240).read()
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _render_tiled(image, vis, bounds, grid=2, dim=TILE_DIM, quality=86):
    """Render an ee.Image above the single-thumbnail size limit by tiling and stitching.

    Earth Engine caps one getThumbURL render (anything much over ~1100 px returns 400), so the
    scene is fetched as a grid of tiles in parallel and composited locally.
    """
    import io
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    w, s, e, n = bounds
    dw, dh = (e - w) / grid, (n - s) / grid
    jobs = []
    for r in range(grid):                       # rows run north -> south
        for c in range(grid):
            tw, te = w + c * dw, w + (c + 1) * dw
            tn, ts = n - r * dh, n - (r + 1) * dh
            url = image.getThumbURL({**vis, "region": ee.Geometry.Rectangle([tw, ts, te, tn]),
                                     "dimensions": dim, "format": "jpg"})
            jobs.append((r, c, url))

    def fetch(job):
        r, c, url = job
        raw = urllib.request.urlopen(url, timeout=300).read()
        return r, c, Image.open(io.BytesIO(raw)).convert("RGB")

    with ThreadPoolExecutor(max_workers=grid * grid) as ex:
        tiles = list(ex.map(fetch, jobs))

    tw_px, th_px = tiles[0][2].size
    out = Image.new("RGB", (tw_px * grid, th_px * grid))
    for r, c, im in tiles:
        if im.size != (tw_px, th_px):
            im = im.resize((tw_px, th_px), Image.LANCZOS)
        out.paste(im, (c * tw_px, r * th_px))

    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), out.size


def _scene_bounds(aoi):
    """Pad the corporation bbox for context and widen it toward a landscape aspect."""
    b = aoi.bounds().coordinates().getInfo()[0]
    w, s, e, n = b[0][0], b[0][1], b[2][0], b[2][1]
    dw, dh = e - w, n - s
    w -= dw * SCENE_MARGIN; e += dw * SCENE_MARGIN
    s -= dh * SCENE_MARGIN; n += dh * SCENE_MARGIN
    dw, dh = e - w, n - s
    if dw / dh < MIN_ASPECT:                      # too portrait -> grow sideways
        need = MIN_ASPECT * dh - dw
        w -= need / 2; e += need / 2
    return [w, s, e, n]


@app.middleware("http")
async def _revalidate_assets(request, call_next):
    """Force the browser to revalidate app code. ETags make this cheap (304s), and it prevents
    a stale cached app.js from silently shadowing a deployed fix."""
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


SCENE_CACHE = "data/processed/scene_cache.json"


def _scene_cache_key(bounds):
    return f"{[round(b, 5) for b in bounds]}|{SAT_GRID}x{TILE_DIM}|{LST_DIM}|{LST_MIN}-{LST_MAX}"


_TILE_LOCK = threading.Lock()


def _mint_tiles():
    """(Re-)sign the Earth Engine tile URL for the thermal layer and record when."""
    with _TILE_LOCK:
        mapid = S["lst_c"].getMapId({"min": LST_MIN, "max": LST_MAX, "palette": PALETTE})
        S["lst_tiles"] = mapid["tile_fetcher"].url_format
        S["lst_tiles_at"] = time.time()
    return S["lst_tiles"]


def _tiles_url():
    """Current tile URL, re-minted if it is close to expiry."""
    if "lst_tiles" not in S or time.time() - S.get("lst_tiles_at", 0) > TILE_TTL_S:
        return _mint_tiles()
    return S["lst_tiles"]


def _stretch_lst_range(lst_c, aoi):
    """Derive the colour range from the composite's own 5th-95th percentile over the city.

    Measured for Kochi: the middle half of all land pixels spans only ~2 degC (36.6-38.6), so any
    wide range paints most of the city one colour. 5th-95th (~31-41) lets the tails saturate and
    the genuine hot spots stand out; 2nd-98th (29-43) was too gentle to separate wards.

    Rounded outward to whole degrees so the legend reads cleanly. Any failure falls back to the
    module defaults rather than blocking startup — the range is presentation, not analysis.
    """
    try:
        r = lst_c.reduceRegion(reducer=ee.Reducer.percentile([5, 95]), geometry=aoi,
                               scale=60, maxPixels=1e9, bestEffort=True).getInfo()
        lo, hi = r.get("lst_c_p5"), r.get("lst_c_p95")
        if lo is None or hi is None:
            raise ValueError("no percentiles returned")
        lo, hi = math.floor(lo), math.ceil(hi)
        if hi - lo < 6:                      # a very flat scene would make the ramp meaningless
            mid = (lo + hi) / 2
            lo, hi = math.floor(mid - 3), math.ceil(mid + 3)
        return int(lo), int(hi)
    except Exception as exc:
        print(f"Could not derive the colour range ({exc}); using {LST_MIN}-{LST_MAX}")
        return LST_MIN, LST_MAX


def _load_roof_shares():
    """Per-ward measured roof share from scripts/build_roof_share.py, if it has been run."""
    try:
        with open(ROOF_SHARE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        wards = {k: float(v["roof_share"]) for k, v in data.get("wards", {}).items()}
        city = data.get("city_default")
        if wards:
            print(f"Roof share: measured for {len(wards)} wards "
                  f"(city {city}), source {data.get('source')}")
            return wards, (float(city) if city else ROOF_SHARE_DEFAULT), True
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Could not read {ROOF_SHARE_PATH} ({exc})")
    print(f"Roof share: no measurement available, using the {ROOF_SHARE_DEFAULT} assumption "
          f"(run scripts/build_roof_share.py)")
    return {}, ROOF_SHARE_DEFAULT, False


@app.on_event("startup")
def _startup():
    cfg = load_config(CFG_PATH)
    ee.Initialize(project=cfg["gee"]["project_id"])
    aoi = aoi_geometry(cfg)
    dry = cfg["time"]["dry_season_months"]

    col = build_collection(cfg, aoi, cfg["time"]["start"], cfg["time"]["end"])
    col = col.map(lambda im: im.set("month", ee.Image(im).date().get("month"))) \
             .filter(ee.Filter.inList("month", dry))
    S["bounds"] = _scene_bounds(aoi)
    west, south, east, north = S["bounds"]
    region = ee.Geometry.Rectangle([west, south, east, north])

    # Clip to the padded scene, not the corporation, so the thermal layer covers the whole map
    # and a drawn area anywhere on screen returns data.
    feat = col.median().clip(region)
    lst_c = feat.select("lst").subtract(273.15).rename("lst_c")
    S["feat"] = feat.addBands(lst_c)
    S["aoi"] = aoi

    S["s2"] = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(region).filterDate("2024-01-01", cfg["time"]["end"])
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15)).median())
    S["lst_c"], S["region"] = lst_c, region

    # Colour range from the data, BEFORE the tile URL is signed: the range is baked into it.
    global LST_MIN, LST_MAX
    LST_MIN, LST_MAX = _stretch_lst_range(lst_c, aoi)
    print(f"Thermal colour range {LST_MIN}-{LST_MAX} degC")

    # Thermal layer as Earth Engine TILES: zoomable with the Mapbox basemap, and it makes
    # startup fast (the stitched static scene is only rendered on demand, as a fallback).
    _mint_tiles()
    print("Earth Engine LST tile layer ready")

    # Data vintage — how many scenes went into the baseline and over what window. A public-sector
    # tool that shows a number without saying how old it is invites misplaced confidence.
    try:
        S["vintage"] = {
            "n_scenes": int(col.size().getInfo()),
            "start": cfg["time"]["start"], "end": cfg["time"]["end"],
            "months": dry,
            "sensor": "Landsat 8/9 C2 L2 (ST_B10), 30 m",
        }
    except Exception as exc:
        print(f"Could not read scene count ({exc})")
        S["vintage"] = {"n_scenes": None, "start": cfg["time"]["start"],
                        "end": cfg["time"]["end"], "months": dry}

    S["model"] = PinnModel()

    # Keep the PER-PIXEL table. The model is non-linear, so averaging features and predicting
    # once (f(mean x)) is not the same as predicting per pixel and averaging (mean f(x)) —
    # Jensen's inequality. A half-park/half-concrete ward analysed as "uniformly semi-built"
    # understates the benefit of treating the built half. All analysis runs per pixel.
    df = pd.read_parquet(PARQUET)
    df = df[df.ward_id.notna()].reset_index(drop=True)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    S["px"] = df
    S["px_by_ward"] = {wid: g.index.to_numpy() for wid, g in df.groupby("ward_id")}

    S["roof_shares"], S["roof_share_city"], S["roof_measured"] = _load_roof_shares()

    dates = pd.to_datetime(df["date"])
    S["vintage"].update(n_pixels=int(len(df)),
                        n_observation_dates=int(dates.dt.date.nunique()),
                        first_date=str(dates.min().date()), last_date=str(dates.max().date()))

    agg = df.groupby("ward_id")[INPUT_COLUMNS + ["lst"]].mean()
    agg["npix"] = df.groupby("ward_id").size()
    S["wards"] = agg
    S["wards_geo"] = _ward_geojson(agg)

    S["metrics"] = _model_metrics(S["model"].name)
    print(f"Planner ready - model {S['model'].name}, {len(agg)} wards, "
          f"{len(df):,} pixels, RMSE {S['metrics'].get('rmse', float('nan')):.2f} K")


def _model_metrics(ckpt_name):
    """Test-set error of the deployed checkpoint, so the UI can show honest uncertainty."""
    path = "docs/figures/pinn_results.json"
    lam = (ckpt_name.split("lambda")[-1].replace(".pt", "") if "lambda" in ckpt_name else None)
    try:
        res = json.load(open(path, encoding="utf-8"))
        for name, m in res.items():
            if lam and f"lam={lam}" in name.replace(" ", ""):
                return {"rmse": float(m["rmse"]), "r2": float(m["r2"]),
                        "seb_residual": float(m.get("abs_residual", float("nan")))}
    except Exception:
        pass
    return {"rmse": float("nan"), "r2": float("nan")}


def _render_scene(s2, lst_c, region, key):
    print(f"Rendering satellite scene ({SAT_GRID}x{SAT_GRID} tiles) ...")
    try:
        S["sat"], size = _render_tiled(
            s2, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.15},
            S["bounds"], grid=SAT_GRID)
        print(f"  satellite {size[0]}x{size[1]} px")
    except Exception as exc:                      # never let detail cost us the whole scene
        print(f"  tiled render failed ({exc}); falling back to a single tile")
        S["sat"] = _fetch_b64(s2.getThumbURL(
            {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.15,
             "region": region, "dimensions": TILE_DIM, "format": "jpg"}), "image/jpeg")

    print("Rendering thermal overlay ...")
    S["lst"] = _fetch_b64(lst_c.getThumbURL(
        {"min": LST_MIN, "max": LST_MAX, "palette": PALETTE,
         "region": region, "dimensions": LST_DIM, "format": "png"}))

    try:
        os.makedirs(os.path.dirname(SCENE_CACHE), exist_ok=True)
        with open(SCENE_CACHE, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "sat": S["sat"], "lst": S["lst"]}, fh)
        print("Scene cached to disk")
    except Exception as exc:
        print(f"Could not cache scene ({exc})")


def _display_names(parts):
    """Planner-facing name for every ward.

    Six polygons in the DataMeet source carry no name or number (their ids are synthetic: x19,
    x35, ...). Showing "Ward x35" to a planner is meaningless, and inventing a number would be
    worse. They are named by their nearest labelled neighbour — "Unnamed ward near Kalvathi" — which
    is true, locatable, and obviously provisional. ward_id is untouched, so exports and saved
    scenarios stay stable. Replace with the official KMC names when they are available.
    """
    named = [(f["properties"]["ward_name"].strip()[:1].upper() + f["properties"]["ward_name"].strip()[1:], g)
             for f, g in parts if f["properties"].get("ward_name")]
    out, used = {}, {}
    for f, g in parts:
        p = f["properties"]
        wid = p["ward_id"]
        if p.get("ward_name"):
            nm = p["ward_name"].strip()
            out[wid] = {"name": nm[:1].upper() + nm[1:], "unnamed": False}   # source has "kaloor South"
            continue
        if named:
            near = min(named, key=lambda ng: (g.distance(ng[1]),
                                              g.centroid.distance(ng[1].centroid)))[0]
            base = f"Unnamed ward near {near}"
        else:
            base = f"Unnamed ward {wid}"
        used[base] = used.get(base, 0) + 1
        out[wid] = {"name": base if used[base] == 1 else f"{base} ({used[base]})", "unnamed": True}
    return out


def _ward_geojson(agg):
    from shapely.geometry import shape, mapping
    gj = json.load(open(WARDS_PATH, encoding="utf-8"))
    parts = [(f, shape(f["geometry"]).simplify(SIMPLIFY_TOL, preserve_topology=True))
             for f in gj["features"]]
    names = _display_names(parts)
    feats = []
    for f, geom in parts:
        wid = f["properties"]["ward_id"]
        has = wid in agg.index
        p = {"ward_id": wid,
             "ward_name": names[wid]["name"],
             "unnamed": names[wid]["unnamed"],
             "ward_no": f["properties"].get("ward_no"),
             "area_sqkm": f["properties"].get("area_sqkm"),
             "has_data": bool(has)}
        if has:
            r = agg.loc[wid]
            p.update(lst_c=round(float(r.lst - 273.15), 2),
                     ndvi=round(float(r.ndvi), 3),
                     albedo=round(float(r.albedo), 3),
                     built_frac=round(float(built_fraction(r.ndvi)), 2),
                     roof_share=round(float(S["roof_shares"].get(
                         wid, S["roof_share_city"])), 3),
                     npix=int(r.npix))
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": p})
    return {"type": "FeatureCollection", "features": feats}


@app.get("/api/scene")
def scene():
    """Static stitched scene — only used when the browser cannot run Mapbox GL (no WebGL).

    Rendered lazily on first request (it costs minutes) and cached to disk afterwards.
    """
    if "sat" not in S:
        key = _scene_cache_key(S["bounds"])
        if os.path.exists(SCENE_CACHE):
            try:
                cached = json.load(open(SCENE_CACHE, encoding="utf-8"))
                if cached.get("key") == key:
                    S["sat"], S["lst"] = cached["sat"], cached["lst"]
                    print("Scene loaded from cache")
            except Exception as exc:
                print(f"Scene cache unreadable ({exc}); re-rendering")
    if "sat" not in S:
        _render_scene(S["s2"], S["lst_c"], S["region"], _scene_cache_key(S["bounds"]))
    return {"sat": S["sat"], "lst": S["lst"], "bounds": S["bounds"],
            "lst_range": [LST_MIN, LST_MAX], "palette": ["#" + c for c in PALETTE]}


@app.get("/api/wards")
def wards():
    return S["wards_geo"]


@app.get("/api/meta")
def meta():
    """Client bootstrap. The Mapbox token is a PUBLIC (pk.) token, intended for browser use —
    it is read from .env (gitignored) rather than committed, and should also be URL-restricted
    in the Mapbox account settings."""
    rmse = S["metrics"].get("rmse")
    r2 = S["metrics"].get("r2")
    n_inert = sum(1 for f in S["wards_geo"]["features"]
                  if not f["properties"]["has_data"])
    return {"model": S["model"].name, "n_wards": int(len(S["wards"])),
            "n_wards_total": len(S["wards_geo"]["features"]),
            "n_wards_no_data": n_inert,
            "no_data_reason": NO_DATA_REASON,
            "city": "Kochi, Kerala", "lst_range": [LST_MIN, LST_MAX],
            "palette": ["#" + c for c in PALETTE],
            "bounds": S["bounds"],
            "lst_tiles": _tiles_url(),
            "tile_ttl_s": TILE_TTL_S,
            "vintage": S.get("vintage", {}),
            "model_rmse": None if rmse is None or rmse != rmse else round(rmse, 2),
            "model_r2": None if r2 is None or r2 != r2 else round(r2, 3),
            "roof_share_default": S["roof_share_city"],
            "roof_share_measured": S["roof_measured"],
            "mapbox_token": MAPBOX_TOKEN}


@app.post("/api/refresh_tiles")
@app.get("/api/refresh_tiles")
def refresh_tiles():
    """Re-sign the Earth Engine tile URL.

    The client calls this when tiles start failing. Without it an expired URL just stops
    rendering the thermal layer, with nothing in the UI to say why.
    """
    return {"lst_tiles": _mint_tiles(), "ttl_s": TILE_TTL_S}


class Interventions(BaseModel):
    albedo_set: float | None = None       # cool roof target albedo
    ndvi_delta: float | None = None       # greening / canopy, NDVI increase
    roof_share: float | None = None       # fraction of built area actually treated


class Selection(BaseModel):
    kind: str                              # "wards" | "bbox" | "polygon"
    ward_ids: list[str] | None = None
    bounds: list[float] | None = None      # [west, south, east, north]
    coordinates: list[list[float]] | None = None   # polygon ring, [[lon, lat], ...]


class AnalyzeRequest(BaseModel):
    selection: Selection
    interventions: Interventions
    season: str | None = None              # "all" | "dry" | "premonsoon"


SEASONS = {"all": None, "dry": [12, 1, 2], "premonsoon": [3, 4]}


def _roof_share_for(frame, iv):
    """Per-pixel roof share: an explicit user override, else the MEASURED per-ward value.

    Returns (array, label). The label says which source was used, so the UI and the CSV export
    can state whether a headline number rests on a measurement or on an assumption.
    """
    if iv.roof_share is not None:
        return np.full(len(frame), float(iv.roof_share)), "user"
    city = S["roof_share_city"]
    if S["roof_measured"] and "ward_id" in frame:
        shares = S["roof_shares"]
        return (frame["ward_id"].map(lambda w: shares.get(w, city))
                .to_numpy(dtype="float64"), "measured")
    return np.full(len(frame), city), ("measured_city" if S["roof_measured"] else "assumed")


def _pixel_response(frame, iv):
    """Run the PINN on EVERY pixel and return per-pixel arrays (no feature averaging)."""
    feats = {c: frame[c].to_numpy(dtype="float64") for c in INPUT_COLUMNS}
    r = whatif(S["model"], feats,
               albedo_set=iv.albedo_set if iv.albedo_set else None,
               ndvi_delta=iv.ndvi_delta if iv.ndvi_delta else None)
    share, share_src = _roof_share_for(frame, iv)
    bf = r["built_frac"]
    # A cool roof only covers the roofs; greening is applied across the built surface.
    coverage = bf * share if iv.albedo_set else bf
    return {
        "t_base": frame["lst"].to_numpy(dtype="float64") - 273.15,   # measured
        "delta_t": r["delta_t"] * coverage,                           # coverage-scaled
        "delta_t_full": r["delta_t"],
        "built_frac": bf,
        "coverage": coverage,
        "roof_share": share,
        "roof_share_source": share_src,
        "seb_residual": r["seb_residual"],
    }


def _drawn_region(sel):
    """Earth Engine geometry for a drawn selection, plus its centre (lon, lat).

    Planners do not work in rectangles, so an arbitrary polygon is accepted alongside the
    rubber-band box. Both go down the same sampling path.
    """
    if sel.kind == "bbox":
        if not sel.bounds or len(sel.bounds) != 4:
            raise HTTPException(400, "bbox selection needs bounds [w,s,e,n]")
        w, s, e, n = sel.bounds
        return ee.Geometry.Rectangle([w, s, e, n]), ((w + e) / 2, (s + n) / 2)

    ring = sel.coordinates or []
    if len(ring) < 3:
        raise HTTPException(400, "polygon selection needs at least 3 coordinates")
    if any(len(pt) != 2 for pt in ring):
        raise HTTPException(400, "polygon coordinates must be [lon, lat] pairs")
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    cx = sum(p[0] for p in ring[:-1]) / (len(ring) - 1)
    cy = sum(p[1] for p in ring[:-1]) / (len(ring) - 1)
    return ee.Geometry.Polygon([ring]), (cx, cy)


def _region_area_sqkm(region):
    """True area of a drawn geometry [km^2], measured by Earth Engine."""
    try:
        return float(region.area(maxError=1).getInfo()) / 1e6
    except Exception as exc:
        raise HTTPException(502, f"could not measure the drawn area ({exc})")


def _season_frame(season):
    df = S["px"]
    months = SEASONS.get(season or "all")
    return df if not months else df[df.month.isin(months)]


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    sel, iv = req.selection, req.interventions
    no_change = not (iv.albedo_set or iv.ndvi_delta)
    geo = {f["properties"]["ward_id"]: f["properties"] for f in S["wards_geo"]["features"]}

    if sel.kind == "wards":
        ids = [w for w in (sel.ward_ids or []) if w in S["px_by_ward"]]
        if not ids:
            raise HTTPException(400, "no selected wards have measured data")
        frame = _season_frame(req.season)
        frame = frame[frame.ward_id.isin(ids)]
        if frame.empty:
            raise HTTPException(422, "no pixels for that season in the selected wards")

        out = _pixel_response(frame, iv)                 # per pixel, then aggregate
        res = pd.DataFrame({"ward_id": frame.ward_id.to_numpy(),
                            "t_base": out["t_base"], "delta_t": out["delta_t"],
                            "delta_t_full": out["delta_t_full"],
                            "built_frac": out["built_frac"],
                            "coverage": out["coverage"],
                            "roof_share": out["roof_share"]})
        g = res.groupby("ward_id").mean(numeric_only=True)
        counts = res.groupby("ward_id").size()
        spread = res.groupby("ward_id")["delta_t"].quantile([0.1, 0.9]).unstack()

        items = []
        for wid in ids:
            if wid not in g.index:
                continue
            r, p = g.loc[wid], geo.get(wid, {})
            a = p.get("area_sqkm")
            items.append({"ward_id": wid, "name": p.get("ward_name", wid), "area_sqkm": a,
                          "t_base": round(float(r.t_base), 2),
                          "t_new": round(float(r.t_base + r.delta_t), 2),
                          "delta_t": round(float(r.delta_t), 2),
                          "delta_t_full": round(float(r.delta_t_full), 2),
                          "delta_t_p10": round(float(spread.loc[wid, 0.1]), 2),
                          "delta_t_p90": round(float(spread.loc[wid, 0.9]), 2),
                          "built_frac": round(float(r.built_frac), 2),
                          "roof_share": round(float(r.roof_share), 3),
                          "treated_sqkm": round(float((a or 0) * r.coverage), 4),
                          "npix": int(counts.loc[wid])})
        area = float(np.nansum([i["area_sqkm"] or 0 for i in items]))
        all_dt = res["delta_t"].to_numpy(dtype="float64")

    elif sel.kind in ("bbox", "polygon"):
        region, centre = _drawn_region(sel)
        # Sample ACTUAL pixels rather than reducing to a mean feature vector first.
        fc = S["feat"].sample(region=region, scale=30, numPixels=DRAW_SAMPLE_PIXELS,
                              seed=1, dropNulls=True, geometries=False).getInfo()
        rows = [f["properties"] for f in fc.get("features", [])]
        rows = [r for r in rows if r.get("lst") is not None]
        if not rows:
            raise HTTPException(
                422, "No usable pixels in that area. It may be open water or cloud, or a coastal "
                     "area the climate inputs do not cover (parts of southern and western Kochi).")

        frame = pd.DataFrame(rows)
        for c in INPUT_COLUMNS:
            if c not in frame:
                frame[c] = {"lon": centre[0], "lat": centre[1]}.get(c, 0.0)
        frame = frame.dropna(subset=[c for c in INPUT_COLUMNS if c in frame] + ["lst"])
        out = _pixel_response(frame, iv)

        # The TRUE area of the drawn shape, from the geometry. It must not be derived from the
        # sample size: the sample is capped at DRAW_SAMPLE_PIXELS, so counting sampled pixels
        # silently clamped every large drawn area to the same few hundred metres square.
        area = _region_area_sqkm(region)
        all_dt = out["delta_t"]
        label = "Drawn area" if sel.kind == "bbox" else "Drawn zone"
        items = [{"ward_id": "area", "name": label, "area_sqkm": round(area, 3),
                  "t_base": round(float(np.mean(out["t_base"])), 2),
                  "t_new": round(float(np.mean(out["t_base"] + out["delta_t"])), 2),
                  "delta_t": round(float(np.mean(out["delta_t"])), 2),
                  "delta_t_full": round(float(np.mean(out["delta_t_full"])), 2),
                  "delta_t_p10": round(float(np.percentile(all_dt, 10)), 2),
                  "delta_t_p90": round(float(np.percentile(all_dt, 90)), 2),
                  "built_frac": round(float(np.mean(out["built_frac"])), 2),
                  "roof_share": round(float(np.mean(out["roof_share"])), 3),
                  "treated_sqkm": round(float(area * np.mean(out["coverage"])), 4),
                  "npix": int(len(frame))}]
    else:
        raise HTTPException(400, f"unknown selection kind '{sel.kind}'")

    dts = np.array([i["delta_t"] for i in items], dtype="float64")
    weights = np.array([(i["area_sqkm"] or 0) for i in items], dtype="float64")
    wsum = weights.sum()
    mean_dt = float(np.average(dts, weights=weights) if wsum else dts.mean())
    rmse = S["metrics"].get("rmse")

    # BUDGET / COST-EFFECTIVENESS.
    #   treated_area_sqkm  the surface actually coated or planted (built fraction x roof share),
    #                      which is what an intervention is costed on.
    #   cooling_k_km2      total cooling delivered, sum(dT_i * area_i) in K.km^2. An extensive
    #                      quantity: a big ward cooled a little can beat a small ward cooled a lot.
    #   cooling_per_treated_km2  the ratio of the two — K.km^2 of cooling bought per km^2 treated.
    #                      This is the figure to compare strategies on, because it is independent
    #                      of how much area a selection happens to contain.
    # (The previous `cooling_per_km2` was mean_dT * area, which is K.km^2 — a TOTAL, despite the
    #  name. It is kept below under the correct name and no longer labelled "per km2".)
    treated = float(np.nansum([i.get("treated_sqkm") or 0 for i in items]))
    cooling_k_km2 = float(np.nansum([(i["delta_t"] * (i["area_sqkm"] or 0)) for i in items]))

    # UNCERTAINTY. Two different things, deliberately not combined:
    #   spread  p10-p90 of dT ACROSS PIXELS — real spatial heterogeneity within the selection.
    #   rmse    the deployed checkpoint's held-out error on absolute LST. dT is a difference of
    #           two predictions from the same model, so this over-states the error on dT, but it
    #           is the honest published figure and is labelled as model error, not as a dT band.
    p10 = float(np.percentile(all_dt, 10)) if len(all_dt) else float("nan")
    p90 = float(np.percentile(all_dt, 90)) if len(all_dt) else float("nan")

    shares = np.array([i.get("roof_share", 0) for i in items], dtype="float64")
    return {
        "items": sorted(items, key=lambda x: x["delta_t"]),
        "summary": {
            "n": len(items),
            "area_sqkm": round(float(area), 2),
            "treated_area_sqkm": round(treated, 3),
            "mean_t_base": round(float(np.mean([i["t_base"] for i in items])), 2),
            "mean_delta_t": round(mean_dt, 2),
            "delta_t_p10": round(p10, 2),
            "delta_t_p90": round(p90, 2),
            "best_delta_t": round(float(dts.min()), 2),
            "best_name": items[int(np.argmin(dts))]["name"],
            "cooling_k_km2": round(cooling_k_km2, 3),
            "cooling_per_treated_km2": round(cooling_k_km2 / treated, 2) if treated else None,
            "no_change": no_change,
            "model": S["model"].name,
            "model_rmse": None if rmse is None or rmse != rmse else round(rmse, 2),
            "season": req.season or "all",
            "roof_share": round(float(shares.mean()), 3) if len(shares) else None,
            "roof_share_source": out["roof_share_source"],
            "n_pixels": int(len(all_dt)),
        },
    }


@app.post("/api/rank")
def rank(req: AnalyzeRequest):
    """Prioritisation: score EVERY ward, so a planner can answer 'where do we start?'.

    Three orderings are returned per ward rather than one composite score, because they answer
    genuinely different questions and a single blended number would hide which one is driving
    the ranking:

        t_base                   where is it hottest now (severity)
        delta_t                  where does this intervention cool most (effectiveness)
        cooling_per_treated_km2  how much cooling per km2 of roof actually treated
                                 (cost-effectiveness — the one a budget conversation needs)
    """
    iv = req.interventions
    frame = _season_frame(req.season)
    out = _pixel_response(frame, iv)
    res = pd.DataFrame({"ward_id": frame.ward_id.to_numpy(),
                        "t_base": out["t_base"], "delta_t": out["delta_t"],
                        "built_frac": out["built_frac"],
                        "coverage": out["coverage"], "roof_share": out["roof_share"]})
    g = res.groupby("ward_id").mean(numeric_only=True)
    counts = res.groupby("ward_id").size()
    geo = {f["properties"]["ward_id"]: f["properties"] for f in S["wards_geo"]["features"]}

    rows = []
    for wid, r in g.iterrows():
        p = geo.get(wid, {})
        area = p.get("area_sqkm") or 0.0
        treated = float(area * r.coverage)
        delivered = float(r.delta_t * area)
        rows.append({"ward_id": wid, "name": p.get("ward_name", wid),
                     "area_sqkm": p.get("area_sqkm"),
                     "t_base": round(float(r.t_base), 2),
                     "delta_t": round(float(r.delta_t), 2),
                     "built_frac": round(float(r.built_frac), 2),
                     "roof_share": round(float(r.roof_share), 3),
                     "treated_sqkm": round(treated, 4),
                     "cooling_k_km2": round(delivered, 3),
                     "cooling_per_treated_km2": (round(delivered / treated, 2)
                                                 if treated > 1e-9 else None),
                     "npix": int(counts.loc[wid])})
    rmse = S["metrics"].get("rmse")
    return {"items": rows, "season": req.season or "all",
            "roof_share_source": out["roof_share_source"],
            "model_rmse": None if rmse is None or rmse != rmse else round(rmse, 2)}


@app.post("/api/export")
def export_csv(req: AnalyzeRequest):
    """CSV of the current scenario, for committee papers."""
    from fastapi.responses import PlainTextResponse
    res = analyze(req)
    iv = req.interventions
    sm = res["summary"]
    v = S.get("vintage", {})
    head = (f"# UrbanHeat Planner scenario\n"
            f"# exported,{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# model,{sm['model']}\n"
            f"# model_rmse_K,{sm.get('model_rmse')}\n"
            f"# season,{sm['season']}\n"
            f"# albedo_set,{iv.albedo_set or ''}\n"
            f"# ndvi_delta,{iv.ndvi_delta or ''}\n"
            f"# roof_share,{sm['roof_share']}\n"
            f"# roof_share_source,{sm['roof_share_source']}\n"
            f"# baseline_sensor,{v.get('sensor', 'Landsat 8/9')}\n"
            f"# baseline_window,{v.get('start', '')} to {v.get('end', '')}\n"
            f"# baseline_scenes,{v.get('n_scenes', '')}\n"
            f"# selection_pixels,{sm.get('n_pixels', '')}\n"
            f"# treated_area_sqkm,{sm.get('treated_area_sqkm')}\n"
            f"# cooling_K_km2_total,{sm.get('cooling_k_km2')}\n"
            f"# cooling_per_treated_km2,{sm.get('cooling_per_treated_km2')}\n"
            f"# NOTE baseline LST is measured Landsat; delta_t is model response, coverage-scaled\n"
            f"# NOTE delta_t_p10/p90 are the spread ACROSS PIXELS, not a model error bar\n")
    cols = ["ward_id", "name", "area_sqkm", "treated_sqkm", "npix", "t_base", "t_new",
            "delta_t", "delta_t_p10", "delta_t_p90", "delta_t_full", "built_frac", "roof_share"]
    lines = [",".join(cols)]
    for i in res["items"]:
        lines.append(",".join(f'"{i.get(c, "")}"' if c == "name" else str(i.get(c, ""))
                              for c in cols))
    return PlainTextResponse(head + "\n".join(lines) + "\n", media_type="text/csv")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="planner")
