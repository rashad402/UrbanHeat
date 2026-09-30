"""First vertical slice: one cloud-free Landsat LST map over the Kochi AOI.

This is the "something real to run" milestone (plan §2, §5). It proves the whole chain works:
GEE access -> AOI -> Landsat C2 L2 -> cloud masking -> LST retrieval -> a rendered map.

Run (after `earthengine authenticate`):
    python scripts/first_slice_lst.py
    python scripts/first_slice_lst.py --config configs/data_config.yaml --start 2023-12-01 --end 2024-04-30

Outputs:
    docs/figures/kochi_lst_first_slice.png   (LST map thumbnail)
    prints mean/min/max LST over the AOI.

Requires: earthengine-api, pyyaml. (No GPU, no heavy compute — all processing is server-side.)
"""

import argparse
import json
import os
import urllib.request

import ee
import yaml

# Allow running as a plain script (python scripts/first_slice_lst.py) without installing the package.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_engine.indices import lst_from_landsat_l2  # noqa: E402


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def aoi_geometry(config):
    """Read the AOI polygon from the GeoJSON referenced in the config -> ee.Geometry."""
    with open(config["aoi"]["geojson"]) as f:
        gj = json.load(f)
    coords = gj["features"][0]["geometry"]["coordinates"]
    return ee.Geometry.Polygon(coords)


def mask_landsat_c2_l2(image):
    """Mask cloud / shadow / cirrus / dilated-cloud using the QA_PIXEL bitmask."""
    qa = image.select("QA_PIXEL")
    # Bit 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow.
    mask = (qa.bitwiseAnd(1 << 1).eq(0)
            .And(qa.bitwiseAnd(1 << 2).eq(0))
            .And(qa.bitwiseAnd(1 << 3).eq(0))
            .And(qa.bitwiseAnd(1 << 4).eq(0)))
    return image.updateMask(mask)


def add_lst_celsius(image):
    """Add an 'LST' band in degC from the scaled ST_B10 product."""
    lst_c = lst_from_landsat_l2(image.select("ST_B10"), to_celsius=True).rename("LST")
    return image.addBands(lst_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--start", default=None, help="override start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="override end date YYYY-MM-DD")
    ap.add_argument("--out", default="docs/figures/kochi_lst_first_slice.png")
    args = ap.parse_args()

    config = load_config(args.config)
    project = config["gee"]["project_id"]

    ee.Initialize(project=project)
    print(f"Earth Engine initialized on project: {project}")

    aoi = aoi_geometry(config)

    # Dry-season window by default (fewer clouds); overridable from the CLI.
    start = args.start or "2023-12-01"
    end = args.end or "2024-04-30"

    def prep(collection_id):
        return (ee.ImageCollection(collection_id)
                .filterBounds(aoi)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUD_COVER", config["satellite"]["landsat"]["cloud_pct_max"]))
                .map(mask_landsat_c2_l2)
                .map(add_lst_celsius))

    l8 = prep(config["satellite"]["landsat"]["collection"])
    l9 = prep(config["satellite"]["landsat"]["also"])
    collection = l8.merge(l9)

    n = collection.size().getInfo()
    print(f"Cloud-filtered Landsat scenes {start}..{end} over AOI: {n}")
    if n == 0:
        print("No scenes found — widen the date range with --start/--end or raise cloud_pct_max.")
        return

    lst = collection.select("LST").median().clip(aoi)

    stats = lst.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
        geometry=aoi, scale=30, maxPixels=1e9,
    ).getInfo()
    print("LST over AOI (degC):", {k: round(v, 2) for k, v in stats.items() if v is not None})

    # Render a thumbnail (server-side) and download it — no local raster processing needed.
    palette = ["040274", "0502a3", "235cb1", "307ef3", "269db1", "30c8e2",
               "86e26f", "d7e219", "efe600", "f89e1a", "e93e3a", "911003"]
    url = lst.getThumbURL({
        "min": 22, "max": 42, "palette": palette,
        "region": aoi, "dimensions": 800, "format": "png",
    })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    urllib.request.urlretrieve(url, args.out)
    print(f"Saved LST map -> {args.out}")


if __name__ == "__main__":
    main()
