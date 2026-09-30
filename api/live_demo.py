"""Live demo backend: real satellite imagery + real 30 m Landsat LST + point/area queries.

This is the "real product" slice (plan §11) — it shows what the ward-level preview can't:
  * a satellite basemap with the actual 30 m LST raster overlaid (served as GEE tiles), and
  * click-anywhere queries at an arbitrary radius (default 50 m), returning the real mean LST
    and a preview ΔT for the chosen intervention over exactly those pixels.

The ΔT here is still the first-order SEB stand-in (same as the ward simulator); swap
`seb_delta_t` for the trained PINN when ready. Runs locally against your GEE auth:

    uvicorn api.live_demo:app --port 8000
    # then open http://localhost:8000/simulator_live.html

Cannot be a published Artifact — that sandbox blocks external map tiles (satellite + GEE).
"""

import ee
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_engine.build_dataset import load_config, aoi_geometry, build_collection

CFG = "configs/data_config.yaml"
app = FastAPI(title="PIML-UrbanHeat live demo")
S = {}   # server state populated at startup

# First-order SEB preview (mirrors web/simulator.html deltaT()).
SIGMA, EPS, RHO_CP, RAH = 5.670374419e-8, 0.96, 1204.8, 60.0
ROOF_SHARE, K_ET, CANOPY_BONUS = 0.5, 0.42, 1.35


def _clamp(x, a, b):
    return max(a, min(b, x))


def seb_delta_t(lst_c, ndvi, albedo, s_down, cool, green, canopy):
    lam = 4 * EPS * SIGMA * (lst_c + 273.15) ** 3 + RHO_CP / RAH
    bf = _clamp(1 - _clamp((ndvi - 0.10) / 0.60, 0, 1), 0, 1)
    d = 0.0
    if cool > albedo:
        d += -((cool - albedo) * s_down / lam) * bf * ROOF_SHARE
    if green > 0:
        d += -(K_ET * green * (s_down / lam)) * bf
    if canopy > 0:
        d += -(K_ET * canopy * (s_down / lam)) * bf * CANOPY_BONUS
    return _clamp(d, -6, 0), bf


@app.on_event("startup")
def _startup():
    cfg = load_config(CFG)
    ee.Initialize(project=cfg["gee"]["project_id"])
    aoi = aoi_geometry(cfg)
    dry = cfg["time"]["dry_season_months"]

    col = build_collection(cfg, aoi, cfg["time"]["start"], cfg["time"]["end"])
    col = col.map(lambda im: im.set("month", ee.Image(im).date().get("month"))) \
             .filter(ee.Filter.inList("month", dry))

    # Median composite of all feature bands (already water-masked in process_scene).
    feat = col.median().clip(aoi)
    lst_c = feat.select("lst").subtract(273.15).rename("lst_c")
    S["feat"] = feat.addBands(lst_c)
    S["aoi"] = aoi

    palette = ["313695", "4575b4", "74add1", "abd9e9", "fee090",
               "fdae61", "f46d43", "d73027", "a50026"]
    mapid = lst_c.getMapId({"min": 28, "max": 46, "palette": palette})
    S["tiles"] = mapid["tile_fetcher"].url_format
    S["lst_range"] = [28, 46]
    S["palette"] = ["#" + c for c in palette]

    c = aoi.centroid(1).coordinates().getInfo()
    S["center"] = {"lng": c[0], "lat": c[1]}
    b = aoi.bounds().coordinates().getInfo()[0]
    S["bounds"] = [b[0][0], b[0][1], b[2][0], b[2][1]]
    print(f"Startup done. LST tiles ready. center={S['center']}")


@app.get("/api/config")
def config():
    return {"tiles": S["tiles"], "center": S["center"], "bounds": S["bounds"],
            "lst_range": S["lst_range"], "palette": S["palette"]}


class Query(BaseModel):
    lng: float
    lat: float
    radius: float = 50.0
    cool: float = 0.15
    green: float = 0.0
    canopy: float = 0.0


@app.post("/api/query")
def query(q: Query):
    region = ee.Geometry.Point([q.lng, q.lat]).buffer(q.radius)
    stats = S["feat"].reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        geometry=region, scale=30, maxPixels=1e8,
    ).getInfo()

    lst_c = stats.get("lst_c_mean")
    if lst_c is None:
        return {"ok": False, "reason": "No valid land pixels here (water, cloud, or outside the city)."}

    def val(key, default):          # treat an existing-but-None band value as missing
        v = stats.get(key)
        return default if v is None else v

    ndvi = val("ndvi_mean", 0.3)
    ndbi = val("ndbi_mean", 0.0)
    albedo = val("albedo_mean", 0.13)
    s_down = val("s_down_mean", 642.0)
    npix = int(val("lst_c_count", 0) or val("lst_count", 0))

    dT, bf = seb_delta_t(lst_c, ndvi, albedo, s_down, q.cool, q.green, q.canopy)
    return {
        "ok": True,
        "lst_c": round(lst_c, 2), "after_c": round(lst_c + dT, 2), "delta_t": round(dT, 2),
        "ndvi": round(ndvi, 3), "ndbi": round(ndbi, 3), "albedo": round(albedo, 3),
        "built_frac": round(bf, 2), "npix": npix,
        "area_m2": round(3.14159 * q.radius ** 2),
    }


# Serve the web/ folder (simulator_live.html, ward_data.js, ...) at the root.
app.mount("/", StaticFiles(directory="web", html=True), name="web")
