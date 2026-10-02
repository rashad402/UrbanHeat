"""Dataset assembly -> the canonical training table (plan §5.5).

Usage (as a module — the package uses relative imports, so running the file directly fails):
    python -m data_engine.build_dataset --config configs/data_config.yaml
    python -m data_engine.build_dataset --per-scene 3000 --max-scenes 12

Pipeline (per Landsat 8/9 C2 L2 scene over the Kochi AOI, dry season):
    cloud-mask (QA_PIXEL) -> scale SR -> NDVI/NDBI/albedo -> LST[K] from ST_B10
    -> attach ERA5-Land climate (t_air[K], rh[%], wind[m/s], s_down[W/m^2]) at overpass time
    -> add lon/lat -> random-sample N valid pixels -> pull via getInfo.
Then locally: join ward_id (shapely point-in-polygon), drop physical outliers, write parquet.

Output columns (DATASET CONTRACT):
    lon, lat, date, ndvi, ndbi, albedo, s_down, t_air, rh, wind, lst, lst_source, ward_id,
    climate_filled
(temperatures in KELVIN; lst_source is 'landsat' in v1 — MODIS gap-fill is a later augmentation.
climate_filled is 1 where the ERA5-Land land-sea mask had no value and the forcing was
interpolated from neighbouring cells — see data_engine/era5.py. It is NOT a model input; it is
there so the coastal pixels can be isolated when reporting.)

Notes:
- Indices are computed from the SAME Landsat scene as the LST target (no cross-sensor time
  mismatch). Sentinel-2 fusion is a documented future enhancement.
- Sampling uses getInfo in per-scene batches for immediate results (keep --per-scene <= 5000).
  For a larger dataset, switch to Export.table.toDrive (see the commented block in sample_scene).
"""

import argparse
import json

import ee
import pandas as pd

from .cloud_mask import mask_landsat_qa
from .era5 import ee_climate_bands

SCHEMA = ["lon", "lat", "date", "ndvi", "ndbi", "albedo",
          "s_down", "t_air", "rh", "wind", "lst", "lst_source", "ward_id", "climate_filled"]

# Landsat C2 L2 scaling.
SR_SCALE, SR_OFFSET = 0.0000275, -0.2
ST_SCALE, ST_OFFSET = 0.00341802, 149.0

# Physical plausibility bounds for cleaning.
LST_MIN_K, LST_MAX_K = 288.0, 333.0     # ~15..60 degC daytime surface
ALBEDO_MIN, ALBEDO_MAX = 0.02, 0.90


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def aoi_geometry(config):
    with open(config["aoi"]["geojson"]) as f:
        gj = json.load(f)
    geom = gj["features"][0]["geometry"]
    if geom["type"] == "Polygon":
        return ee.Geometry.Polygon(geom["coordinates"])
    return ee.Geometry.MultiPolygon(geom["coordinates"])


def process_scene(img):
    """Cloud-mask a raw Landsat C2 L2 image and return a stacked feature image.

    Water is masked out (MNDWI > 0): this is a LAND surface temperature / urban-heat model,
    and including water (cool, negative NDVI) confounds the feature-LST relationships
    (e.g. it flips the NDVI-LST correlation from the physically expected negative to positive).
    """
    img = mask_landsat_qa(img)
    sr = img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]).multiply(SR_SCALE).add(SR_OFFSET)

    ndvi = sr.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
    ndbi = sr.normalizedDifference(["SR_B6", "SR_B5"]).rename("ndbi")
    mndwi = sr.normalizedDifference(["SR_B3", "SR_B6"]).rename("mndwi")   # >0 => water
    albedo = sr.expression(
        "0.300*B + 0.277*R + 0.233*N + 0.143*S1 + 0.047*S2",
        {"B": sr.select("SR_B2"), "R": sr.select("SR_B4"), "N": sr.select("SR_B5"),
         "S1": sr.select("SR_B6"), "S2": sr.select("SR_B7")},
    ).rename("albedo")
    lst = img.select("ST_B10").multiply(ST_SCALE).add(ST_OFFSET).rename("lst")  # Kelvin

    millis = img.get("system:time_start")
    climate = ee_climate_bands(millis)                       # t_air, rh, wind, s_down
    lonlat = ee.Image.pixelLonLat().rename(["lon", "lat"])

    return (ndvi.addBands([ndbi, albedo, lst, climate, lonlat])
            .updateMask(mndwi.lte(0))                        # keep land only
            .set("system:time_start", millis))


def build_collection(config, aoi, start, end):
    cloud_max = config["satellite"]["landsat"]["cloud_pct_max"]

    def prep(cid):
        return (ee.ImageCollection(cid)
                .filterBounds(aoi).filterDate(start, end)
                .filter(ee.Filter.lt("CLOUD_COVER", cloud_max)))

    raw = prep(config["satellite"]["landsat"]["collection"]).merge(
        prep(config["satellite"]["landsat"]["also"]))
    return raw.map(process_scene)


def sample_scene(feature_img, aoi, per_scene, seed):
    """Random-sample valid pixels from one scene; return a list of property dicts."""
    fc = feature_img.sample(region=aoi, scale=30, numPixels=per_scene,
                            seed=seed, dropNulls=True, geometries=False)
    return fc.getInfo()["features"]
    # --- For a larger dataset, replace the two lines above with an async Drive export:
    # task = ee.batch.Export.table.toDrive(collection=fc, description=..., fileFormat="CSV")
    # task.start()  # then download the CSV from Drive and concat.


def ward_joiner(wards_geojson):
    """Return a function (lon, lat) -> ward_id using a shapely spatial index."""
    from shapely.geometry import shape, Point
    from shapely.strtree import STRtree
    with open(wards_geojson, encoding="utf-8") as f:
        feats = json.load(f)["features"]
    geoms = [shape(f["geometry"]) for f in feats]
    ids = [f["properties"]["ward_id"] for f in feats]
    tree = STRtree(geoms)

    def assign(lon, lat):
        pt = Point(lon, lat)
        for idx in tree.query(pt):          # shapely 2.x: candidate indices by bbox
            if geoms[int(idx)].covers(pt):
                return ids[int(idx)]
        return None

    return assign


def build(config_path, per_scene=None, max_scenes=None):
    config = load_config(config_path)
    ee.Initialize(project=config["gee"]["project_id"])
    print(f"Earth Engine initialized on project: {config['gee']['project_id']}")

    aoi = aoi_geometry(config)
    start, end = config["time"]["start"], config["time"]["end"]
    # Restrict to dry-season months for cleaner (less cloudy) scenes.
    dry = config["time"]["dry_season_months"]
    per_scene = per_scene or config["sampling"].get("per_scene", 3000)
    seed = config.get("seed", 42)

    # Keep only dry-season months (Dec..Apr wraps the year, so tag each scene's month
    # and filter with inList rather than calendarRange).
    col = build_collection(config, aoi, start, end)
    col = col.map(lambda im: im.set("month", ee.Image(im).date().get("month"))) \
             .filter(ee.Filter.inList("month", dry))

    times = col.aggregate_array("system:time_start").getInfo()
    n = len(times)
    if max_scenes:
        n = min(n, max_scenes)
    print(f"Dry-season scenes to sample: {len(times)}"
          + (f" (using first {n})" if max_scenes and max_scenes < len(times) else ""))
    if n == 0:
        print("No scenes found — check AOI/date window/cloud threshold in the config.")
        return

    col_list = col.toList(col.size())
    rows = []
    for i in range(n):
        img = ee.Image(col_list.get(i))
        date = pd.to_datetime(times[i], unit="ms").strftime("%Y-%m-%d")
        feats = sample_scene(img, aoi, per_scene, seed + i)
        for ft in feats:
            p = ft["properties"]
            p["date"] = date
            rows.append(p)
        print(f"  [{i+1}/{n}] {date}: sampled {len(feats)} pixels (cumulative {len(rows)})")

    df = pd.DataFrame(rows)
    raw_n = len(df)

    # --- Clean: physical plausibility ---
    df = df.dropna(subset=["lst", "ndvi", "ndbi", "albedo", "s_down", "t_air", "rh", "wind"])
    df = df[(df.lst >= LST_MIN_K) & (df.lst <= LST_MAX_K)]
    df = df[(df.albedo >= ALBEDO_MIN) & (df.albedo <= ALBEDO_MAX)]
    df = df[(df.ndvi >= -1) & (df.ndvi <= 1) & (df.ndbi >= -1) & (df.ndbi <= 1)]
    df = df[(df.rh >= 0) & (df.rh <= 100) & (df.wind >= 0) & (df.s_down >= 0)]
    print(f"Cleaning: {raw_n} -> {len(df)} rows kept ({raw_n - len(df)} dropped)")

    # --- Ward join (local shapely) ---
    assign = ward_joiner(config["aoi"]["wards_geojson"])
    df["ward_id"] = [assign(lo, la) for lo, la in zip(df.lon, df.lat)]
    in_ward = df.ward_id.notna().sum()
    print(f"Ward join: {in_ward}/{len(df)} pixels fell inside a ward polygon")

    df["lst_source"] = "landsat"
    # Sampling returns the flag as a float; keep it small and integral.
    df["climate_filled"] = df.get("climate_filled", 0).fillna(0).astype("int8")
    n_filled = int(df.climate_filled.sum())
    print(f"Climate gap-fill: {n_filled}/{len(df)} rows ({100 * n_filled / max(len(df), 1):.1f}%) "
          f"use ERA5 forcing interpolated over the coastal land-sea mask")
    df = df[SCHEMA]

    out = config["output"]["samples_file"]
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        df.to_parquet(out, index=False)
    except Exception as e:
        out = out.replace(".parquet", ".csv")
        df.to_csv(out, index=False)
        print(f"(parquet unavailable: {e}; wrote CSV instead)")
    print(f"Wrote {out}  ({len(df)} rows, {df.date.nunique()} dates)")
    print(df[["ndvi", "ndbi", "albedo", "s_down", "t_air", "rh", "wind", "lst"]].describe().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--per-scene", type=int, default=None)
    ap.add_argument("--max-scenes", type=int, default=None)
    args = ap.parse_args()
    build(args.config, per_scene=args.per_scene, max_scenes=args.max_scenes)
