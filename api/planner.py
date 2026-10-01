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
import os
import sys
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
LST_MIN, LST_MAX = 28, 46


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

    # Thermal layer as Earth Engine TILES: zoomable with the Mapbox basemap, and it makes
    # startup fast (the stitched static scene is only rendered on demand, as a fallback).
    mapid = lst_c.getMapId({"min": LST_MIN, "max": LST_MAX, "palette": PALETTE})
    S["lst_tiles"] = mapid["tile_fetcher"].url_format
    print("Earth Engine LST tile layer ready")

    S["model"] = PinnModel()
    df = pd.read_parquet(PARQUET)
    df = df[df.ward_id.notna()]
    agg = df.groupby("ward_id")[INPUT_COLUMNS + ["lst"]].mean()
    agg["npix"] = df.groupby("ward_id").size()
    S["wards"] = agg
    S["wards_geo"] = _ward_geojson(agg)
    print(f"Planner ready - model {S['model'].name}, {len(agg)} wards with data")


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


def _ward_geojson(agg):
    from shapely.geometry import shape, mapping
    gj = json.load(open(WARDS_PATH, encoding="utf-8"))
    feats = []
    for f in gj["features"]:
        wid = f["properties"]["ward_id"]
        geom = shape(f["geometry"]).simplify(SIMPLIFY_TOL, preserve_topology=True)
        has = wid in agg.index
        p = {"ward_id": wid,
             "ward_name": f["properties"].get("ward_name") or f"Ward {wid}",
             "ward_no": f["properties"].get("ward_no"),
             "area_sqkm": f["properties"].get("area_sqkm"),
             "has_data": bool(has)}
        if has:
            r = agg.loc[wid]
            p.update(lst_c=round(float(r.lst - 273.15), 2),
                     ndvi=round(float(r.ndvi), 3),
                     albedo=round(float(r.albedo), 3),
                     built_frac=round(float(built_fraction(r.ndvi)), 2),
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
    return {"model": S["model"].name, "n_wards": int(len(S["wards"])),
            "city": "Kochi, Kerala", "lst_range": [LST_MIN, LST_MAX],
            "palette": ["#" + c for c in PALETTE],
            "bounds": S["bounds"],
            "lst_tiles": S["lst_tiles"],
            "mapbox_token": MAPBOX_TOKEN}


class Interventions(BaseModel):
    albedo_set: float | None = None       # cool roof target albedo
    ndvi_delta: float | None = None       # greening / canopy, NDVI increase


class Selection(BaseModel):
    kind: str                              # "wards" | "bbox"
    ward_ids: list[str] | None = None
    bounds: list[float] | None = None      # [west, south, east, north]


class AnalyzeRequest(BaseModel):
    selection: Selection
    interventions: Interventions


def _run(feats, iv, observed_c, areas=None):
    r = whatif(S["model"], feats,
               albedo_set=iv.albedo_set if iv.albedo_set else None,
               ndvi_delta=iv.ndvi_delta if iv.ndvi_delta else None)
    d_applied = r["delta_t_applied"]
    return {
        "t_base": observed_c,
        "t_new": observed_c + d_applied,
        "delta_t": d_applied,
        "delta_t_full": r["delta_t"],
        "built_frac": r["built_frac"],
        "seb_residual": r["seb_residual"],
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    sel, iv = req.selection, req.interventions
    no_change = not (iv.albedo_set or iv.ndvi_delta)

    if sel.kind == "wards":
        ids = [w for w in (sel.ward_ids or []) if w in S["wards"].index]
        if not ids:
            raise HTTPException(400, "no selected wards have measured data")
        sub = S["wards"].loc[ids]
        feats = {c: sub[c].to_numpy() for c in INPUT_COLUMNS}
        observed = sub.lst.to_numpy() - 273.15
        out = _run(feats, iv, observed)

        geo = {f["properties"]["ward_id"]: f["properties"] for f in S["wards_geo"]["features"]}
        items = []
        for i, wid in enumerate(ids):
            p = geo.get(wid, {})
            items.append({"ward_id": wid, "name": p.get("ward_name", wid),
                          "area_sqkm": p.get("area_sqkm"),
                          "t_base": round(float(out["t_base"][i]), 2),
                          "t_new": round(float(out["t_new"][i]), 2),
                          "delta_t": round(float(out["delta_t"][i]), 2),
                          "delta_t_full": round(float(out["delta_t_full"][i]), 2),
                          "built_frac": round(float(out["built_frac"][i]), 2)})
        area = float(np.nansum([i["area_sqkm"] or 0 for i in items]))

    elif sel.kind == "bbox":
        if not sel.bounds or len(sel.bounds) != 4:
            raise HTTPException(400, "bbox selection needs bounds [w,s,e,n]")
        w, s, e, n = sel.bounds
        region = ee.Geometry.Rectangle([w, s, e, n])
        stats = S["feat"].reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=region, scale=30, maxPixels=1e8).getInfo()
        if stats.get("lst_c_mean") is None:
            raise HTTPException(422, "No land pixels in that area (water, cloud, or outside the city).")

        def v(k, d):
            x = stats.get(k)
            return d if x is None else x

        feats = {"lon": np.array([(w + e) / 2]), "lat": np.array([(s + n) / 2]),
                 "ndvi": np.array([v("ndvi_mean", 0.3)]), "ndbi": np.array([v("ndbi_mean", 0.0)]),
                 "albedo": np.array([v("albedo_mean", 0.13)]),
                 "s_down": np.array([v("s_down_mean", 642.0)]),
                 "t_air": np.array([v("t_air_mean", 301.8)]),
                 "rh": np.array([v("rh_mean", 67.0)]),
                 "wind": np.array([v("wind_mean", 1.2)])}
        observed = np.array([stats["lst_c_mean"]])
        out = _run(feats, iv, observed)
        npix = int(v("lst_c_count", 0))
        area = npix * 900 / 1e6
        items = [{"ward_id": "area", "name": "Drawn area", "area_sqkm": round(area, 3),
                  "t_base": round(float(out["t_base"][0]), 2),
                  "t_new": round(float(out["t_new"][0]), 2),
                  "delta_t": round(float(out["delta_t"][0]), 2),
                  "delta_t_full": round(float(out["delta_t_full"][0]), 2),
                  "built_frac": round(float(out["built_frac"][0]), 2)}]
    else:
        raise HTTPException(400, f"unknown selection kind '{sel.kind}'")

    dts = np.array([i["delta_t"] for i in items], dtype="float64")
    weights = np.array([(i["area_sqkm"] or 0) for i in items], dtype="float64")
    wsum = weights.sum()
    return {
        "items": sorted(items, key=lambda x: x["delta_t"]),
        "summary": {
            "n": len(items),
            "area_sqkm": round(float(area), 2),
            "mean_t_base": round(float(np.mean([i["t_base"] for i in items])), 2),
            "mean_delta_t": round(float(np.average(dts, weights=weights) if wsum else dts.mean()), 2),
            "best_delta_t": round(float(dts.min()), 2),
            "best_name": items[int(np.argmin(dts))]["name"],
            "no_change": no_change,
            "model": S["model"].name,
        },
    }


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="planner")
