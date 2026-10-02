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

    # Rasterising footprints is far cheaper than a per-ward vector intersection, and at 30 m it
    # matches the resolution everything else in the pipeline works at.
    roof = (buildings.map(lambda f: f.set("one", 1))
            .reduceToImage(["one"], ee.Reducer.first())
            .gt(0).unmask(0)
            .multiply(ee.Image.pixelArea()).rename("roof_m2"))

    stack = built_area.addBands(roof)
    stats = stack.reduceRegions(collection=wards, reducer=ee.Reducer.sum(), scale=30).getInfo()

    out, tot_roof, tot_built = {}, 0.0, 0.0
    for f in stats["features"]:
        p = f["properties"]
        wid = p["ward_id"]
        built = float(p.get("built_m2") or 0.0)
        rf = float(p.get("roof_m2") or 0.0)
        tot_roof += rf
        tot_built += built
        if built <= 0:
            continue
        out[wid] = {"roof_share": round(min(rf / built, 1.0), 4),
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


if __name__ == "__main__":
    main()
