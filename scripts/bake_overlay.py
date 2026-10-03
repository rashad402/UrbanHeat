"""Bake the thermal overlay: compute it ONCE, save it as static map tiles.

Usage:
    python scripts/bake_overlay.py                       # export from Earth Engine, write tiles
    python scripts/bake_overlay.py --reuse               # re-tile the last export, no Earth Engine
    python scripts/bake_overlay.py --zmax 14 --range 30 42

WHY
    The overlay is a recipe ("median of 94 scenes"), not an image. Served live, Earth Engine runs
    that recipe tile by tile on every first view, so the layer fills in patch by patch, the signed
    tile URLs expire after 45 minutes, and the app cannot show its main layer without an Earth
    Engine login. The scenes are history; they do not change between page loads. So the median is
    computed here once, cut into a standard {z}/{x}/{y}.png pyramid, and served as plain files.

    Box and Zone still query Earth Engine live: a stored layer cannot give NDVI, albedo and
    climate for an arbitrary shape.

WHEN TO RE-RUN
    When new Landsat scenes should enter the median (monthly is plenty), or when the colour range,
    palette, extent or zoom levels change: colours are baked into the tiles.

OUTPUT   web/planner/tiles/{z}/{x}/{y}.png   and   web/planner/tiles/manifest.json
         data/processed/lst_median.npz      (the raw export, gitignored, for --reuse)
"""

import argparse
import datetime
import io
import json
import os
import shutil
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_engine import tiling                      # noqa: E402
from data_engine.overlay import PALETTE             # noqa: E402

CFG = "configs/data_config.yaml"
OUT = "web/planner/tiles"
RAW = "data/processed/lst_median.npz"
NODATA = -9999.0


def export_from_earth_engine(cfg_path, scale):
    """Download the median composite as a numpy array, with per-pixel longitude and latitude."""
    import ee
    from data_engine.build_dataset import load_config
    from data_engine.overlay import build_composite, stretch_range

    cfg = load_config(cfg_path)
    ee.Initialize(project=cfg["gee"]["project_id"])
    print(f"Earth Engine ready on {cfg['gee']['project_id']}")

    comp = build_composite(cfg)
    bounds = comp["bounds"]
    print(f"Extent  [W {bounds[0]:.4f}  S {bounds[1]:.4f}  E {bounds[2]:.4f}  N {bounds[3]:.4f}]")

    n_scenes = int(comp["col"].size().getInfo())
    print(f"Scenes  {n_scenes} (dry season {cfg['time']['start']} to {cfg['time']['end']})")

    lo_hi = stretch_range(comp["lst_c"], comp["aoi"])
    print(f"Colour range from the city's own 5th-95th percentile: {lo_hi[0]}-{lo_hi[1]} degC")

    # Masked pixels (cloud in every scene, water, outside the clip) become -9999 so they survive the
    # download. The lon/lat bands are what let us recover the grid without trusting a transform.
    image = (comp["lst_c"].unmask(NODATA)
             .addBands(ee.Image.pixelLonLat())
             .select(["lst_c", "longitude", "latitude"]).toFloat())
    region = ee.Geometry.Rectangle(bounds, proj=None, geodesic=False)
    url = image.getDownloadURL({"region": region, "scale": scale, "crs": "EPSG:3857",
                                "format": "NPY"})
    print(f"Downloading the {scale} m median composite (this is the slow step) ...")
    data = urllib.request.urlopen(url, timeout=900).read()
    arr = np.load(io.BytesIO(data))
    print(f"  received {len(data) / 1e6:.1f} MB, grid {arr.shape[1]} x {arr.shape[0]} px")

    vals = arr["lst_c"].astype(np.float32)
    vals[~np.isfinite(vals) | (vals < -9000)] = np.nan
    meta = {"bounds": bounds, "n_scenes": n_scenes, "lst_range": list(lo_hi),
            "start": cfg["time"]["start"], "end": cfg["time"]["end"],
            "months": cfg["time"]["dry_season_months"], "scale_m": scale,
            "collections": [cfg["satellite"]["landsat"]["collection"],
                            cfg["satellite"]["landsat"]["also"]]}
    return vals, arr["longitude"], arr["latitude"], meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=CFG)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--scale", type=float, default=30.0, help="export resolution in metres")
    ap.add_argument("--zmin", type=int, default=8)
    ap.add_argument("--zmax", type=int, default=15,
                    help="beyond this Mapbox enlarges the last level; 15 shows the 30 m pixels as "
                         "crisp blocks")
    ap.add_argument("--range", type=float, nargs=2, metavar=("LO", "HI"),
                    help="override the colour range (default: derived from the data)")
    ap.add_argument("--reuse", action="store_true",
                    help=f"re-tile the last export ({RAW}) without touching Earth Engine")
    args = ap.parse_args()

    if args.reuse:
        z = np.load(RAW, allow_pickle=True)
        vals, lon, lat = z["lst"], z["lon"], z["lat"]
        meta = json.loads(str(z["meta"]))
        print(f"Re-using {RAW}  ({vals.shape[1]} x {vals.shape[0]} px, {meta['n_scenes']} scenes)")
    else:
        vals, lon, lat, meta = export_from_earth_engine(args.config, args.scale)
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        np.savez_compressed(RAW, lst=vals, lon=lon, lat=lat, meta=json.dumps(meta))
        print(f"Saved the raw export to {RAW} (gitignored; use --reuse to re-tile it)")

    raster = tiling.Raster.from_lonlat(vals, lon, lat)         # refuses an irregular or flipped grid
    valid = np.isfinite(raster.values)
    print(f"Grid recovered: {raster.width} x {raster.height} px, "
          f"{raster.native_metres:.1f} m on the ground, {100 * valid.mean():.0f}% valid")
    v = raster.values[valid]
    print(f"Temperature: min {v.min():.1f}  p5 {np.percentile(v, 5):.1f}  "
          f"median {np.median(v):.1f}  p95 {np.percentile(v, 95):.1f}  max {v.max():.1f} degC")

    lo, hi = args.range if args.range else meta["lst_range"]
    bounds = meta["bounds"]

    if os.path.isdir(args.out):                                # a stale pyramid must not linger
        shutil.rmtree(args.out)
    print(f"\nBaking z{args.zmin}-z{args.zmax} into {args.out}")
    stats = tiling.bake_pyramid(raster, bounds, args.out, lo, hi, PALETTE, args.zmin, args.zmax)

    stamp = datetime.datetime.now(datetime.timezone.utc)
    manifest = {
        "version": stamp.strftime("%Y%m%d-%H%M%S"),           # the ?v= on every tile URL
        "baked_at": stamp.isoformat(timespec="seconds"),
        "lst_range": [lo, hi],
        "palette": ["#" + c for c in PALETTE],
        "bounds": bounds,
        "minzoom": args.zmin, "maxzoom": args.zmax, "tile_size": tiling.TILE,
        "n_tiles": stats["tiles"], "bytes": stats["bytes"],
        "grid": {"width": raster.width, "height": raster.height, "scale_m": meta["scale_m"],
                 "valid_fraction": round(float(valid.mean()), 4)},
        "source": {k: meta[k] for k in ("n_scenes", "start", "end", "months", "collections")},
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nDone: {stats['tiles']} tiles, {stats['bytes'] / 1e6:.2f} MB "
          f"({stats['empty']} fully transparent). Manifest version {manifest['version']}.")
    print("Restart the planner to serve them; commit web/planner/tiles/ to ship them.")


if __name__ == "__main__":
    main()
