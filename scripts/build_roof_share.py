"""Measure the roof share of built-up area from building footprints (plan §11).

Usage:
    python scripts/build_roof_share.py
    python scripts/build_roof_share.py --confidence 0.70

WHAT THIS REPLACES
    api/inference.py carried ROOF_SHARE = 0.5, "fraction of built-up area that is roof", as a
    flat guess. That constant multiplies EVERY headline number the planner reports, so a guess
    there is not a detail — it silently sets the scale of the whole answer.

    This script measures it instead, per ward, from the Google Open Buildings v3 footprint
    layer (which covers India at high recall) intersected with the ward boundaries:

        roof_share(ward) = building footprint area / BUILT-UP area

    The denominator is built-up area, not ward area, because that is how the number is used:
    api/planner.py computes coverage = built_frac(NDVI) * roof_share, where built_frac already
    accounts for how much of the ward is built. Dividing by ward area instead would double-count
    the vegetated fraction and understate every result.

OUTPUT
    configs/roof_share_kochi.json — {ward_id: {roof_share, built_sqkm, roof_sqkm, n_buildings}}
    plus a city-wide default for wards with no footprint coverage. api/planner.py loads it when
    present and falls back to the documented constant when it is absent, so the repository stays
    runnable without this step.

HOW ROOF AREA IS MEASURED
    From each footprint's own `area_in_meters`, summed per ward. NOT by rasterising footprints to
    a binary 30 m mask: that counts a whole 900 m2 pixel wherever any building touches it, which
    overstated Kochi's roof area 3x (65.43 against a true 21.68 km2) and pinned the ratio at 1.0
    on 72 of 74 wards.

CAVEAT WORTH KEEPING IN THE REPORT
    Open Buildings has lower recall on small informal structures and on dense low-rise fabric,
    both common in Kochi. The measured roof share is therefore a LOWER BOUND, which makes the
    planner's cooling estimates conservative. The UI exposes the value as an adjustable
    assumption for exactly this reason.
"""

import argparse
import json
import os
import sys

import ee

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_engine.build_dataset import (load_config, aoi_geometry,          # noqa: E402
                                       build_collection)

BUILDINGS = "GOOGLE/Research/open-buildings/v3/polygons"
WARDS_PATH = "configs/wards_kochi.geojson"
OUT_PATH = "configs/roof_share_kochi.json"

# NDVI -> built fraction, identical to api.inference.built_fraction. Duplicated as an Earth
# Engine expression because that function is numpy-side; the two must stay in step.
NDVI_BUILT_LO, NDVI_BUILT_SPAN = 0.10, 0.60


def built_fraction_image(ndvi):
    """Earth Engine mirror of api.inference.built_fraction."""
    return ndvi.subtract(NDVI_BUILT_LO).divide(NDVI_BUILT_SPAN).clamp(0, 1).multiply(-1).add(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--confidence", type=float, default=0.70,
                    help="minimum Open Buildings confidence to count a footprint")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ee.Initialize(project=cfg["gee"]["project_id"])
    aoi = aoi_geometry(cfg)

    # Built-up area per ward, from the same dry-season median composite the model is trained on.
    col = build_collection(cfg, aoi, cfg["time"]["start"], cfg["time"]["end"])
    dry = cfg["time"]["dry_season_months"]
    col = col.map(lambda im: im.set("month", ee.Image(im).date().get("month"))) \
             .filter(ee.Filter.inList("month", dry))
    ndvi = col.median().select("ndvi")
    built_area = built_fraction_image(ndvi).multiply(ee.Image.pixelArea()).rename("built_m2")

    with open(WARDS_PATH, encoding="utf-8") as fh:
        gj = json.load(fh)
    wards = ee.FeatureCollection([
        ee.Feature(ee.Geometry(f["geometry"]), {"ward_id": f["properties"]["ward_id"]})
        for f in gj["features"]])

    buildings = (ee.FeatureCollection(BUILDINGS)
                 .filterBounds(aoi)
                 .filter(ee.Filter.gte("confidence", args.confidence)))
    print(f"footprints with confidence >= {args.confidence}: "
          f"{buildings.size().getInfo():,}")

    # Built-up area is a raster quantity, so it is summed per ward from the image.
    built_stats = built_area.reduceRegions(collection=wards, reducer=ee.Reducer.sum(),
                                           scale=30).getInfo()
    built_by_ward = {f["properties"]["ward_id"]: float(f["properties"].get("sum") or 0.0)
                     for f in built_stats["features"]}

    # Roof area is a VECTOR quantity and is summed as one, from each footprint's own
    # area_in_meters. This replaced a 30 m binary rasterisation, which marked a pixel as roof if
    # any building touched it and then counted the whole 900 m2 pixel: a 60 m2 house became
    # 900 m2. In Kochi's dense fabric nearly every built pixel is touched by something, so the
    # city total came out at 65.43 km2 against a true 21.68 km2 (3x), the ratio pinned at 1.0 on
    # 72 of 74 wards, and the min(..., 1.0) below silently made it look merely saturated rather
    # than wrong. Per-ward calls are slower than one raster pass, but this runs once.
    # Wards whose ratio exceeded 1.0 before clamping. More roof than built-up area is physically
    # impossible, so anything here means a real problem upstream, not a saturated ward.
    over = []
    out, tot_roof, tot_built = {}, 0.0, 0.0
    feats = gj["features"]
    for i, f in enumerate(feats, 1):
        wid = f["properties"]["ward_id"]
        geom = ee.Geometry(f["geometry"])
        # filterBounds keeps any footprint intersecting the ward, so a building straddling a
        # boundary is counted in both. At ~10 m buildings against ~1 km wards that is negligible.
        rf = float(buildings.filterBounds(geom).aggregate_sum("area_in_meters").getInfo() or 0.0)
        built = built_by_ward.get(wid, 0.0)
        tot_roof += rf
        tot_built += built
        print(f"  [{i:>2}/{len(feats)}] {wid:<6} roof {rf / 1e6:7.4f} km2   "
              f"built {built / 1e6:7.4f} km2", flush=True)
        if built <= 0:
            continue
        raw = rf / built
        if raw > 1.0:
            over.append((wid, round(raw, 3)))
        out[wid] = {"roof_share": round(min(raw, 1.0), 4),
                    "built_sqkm": round(built / 1e6, 4),
                    "roof_sqkm": round(rf / 1e6, 4)}

    city = round(min(tot_roof / tot_built, 1.0), 4) if tot_built else None
    payload = {"source": BUILDINGS, "confidence": args.confidence,
               "definition": "building footprint area / built-up area (NDVI-derived)",
               "city_default": city, "wards": out}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    shares = sorted(v["roof_share"] for v in out.values())
    if shares:
        mid = shares[len(shares) // 2]
        print(f"\nWrote {args.out}")
        print(f"  {len(out)} wards | city-wide roof share {city:.3f} | "
              f"ward range {shares[0]:.3f}-{shares[-1]:.3f}, median {mid:.3f}")
        print(f"  (the hard-coded guess this replaces was 0.50)")
    else:
        print("No wards had built-up area — check the AOI and ward file.")

    # Loud on purpose. More roof than built-up area is physically impossible, so it means the
    # measurement is wrong rather than saturated - which is exactly what the silent clamp hid.
    if over:
        print(f"\n  WARNING: {len(over)} ward(s) had MORE roof than built-up area before "
              f"clamping to 1.0.")
        print(f"  That is physically impossible; the measurement is wrong, not merely saturated.")
        print(f"  Worst: {sorted(over, key=lambda t: -t[1])[:5]}")


if __name__ == "__main__":
    main()
