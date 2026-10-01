"""Export DENSE image chips from Earth Engine for the CNN baseline (plan §5.1, §7).

Usage:
    python scripts/export_chips.py --patch 9 --per-scene 1200
    python scripts/export_chips.py --patch 13 --per-scene 800 --max-scenes 10

WHY THIS EXISTS
    data_engine/chips.py can rasterise the existing pixel table back onto its 30 m grid, but
    that table is a ~4% random SAMPLE of the AOI, so a 9x9 window reconstructed from it contains
    only a handful of real observations and the rest is fill. A CNN fed mostly fill cannot learn
    neighbourhood structure, and a baseline built that way would understate the CNN unfairly.

    This script instead asks Earth Engine for the neighbourhood directly:
    `ee.Image.neighborhoodToArray` attaches the full patch x patch window of every band to each
    sampled point, so each chip is 100% real observations.

CONTRACT
    Writes data/processed/kochi_chips_p{patch}.npz with
        img      [N, C, patch, patch] float32, channels = data_engine.chips.SPATIAL_CHANNELS
        scalar   [N, S] float32, columns = data_engine.chips.SCALAR_CHANNELS
        y        [N] float32, centre-pixel LST [K]
        lon/lat  [N] float32, chip centre — used for the spatial-block split
        date     [N] str
    scripts/run_cnn.py reads this directly with --chips.

The spatial-block split is applied by the consumer on the chip CENTRES, and with a buffer of
patch//2 pixels so that no test chip's window reaches into a training block.
"""

import argparse
import os
import sys

import ee
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_engine.build_dataset import (load_config, aoi_geometry,          # noqa: E402
                                       build_collection)
from data_engine.chips import SPATIAL_CHANNELS, SCALAR_CHANNELS            # noqa: E402

OUT_DIR = "data/processed"
SCALE = 30


def export(config_path="configs/data_config.yaml", patch=9, per_scene=1200,
           max_scenes=None, seed=42, out=None):
    cfg = load_config(config_path)
    ee.Initialize(project=cfg["gee"]["project_id"])
    aoi = aoi_geometry(cfg)

    col = build_collection(cfg, aoi, cfg["time"]["start"], cfg["time"]["end"])
    dry = cfg["time"]["dry_season_months"]
    col = col.map(lambda im: im.set("month", ee.Image(im).date().get("month"))) \
             .filter(ee.Filter.inList("month", dry))
    ids = col.aggregate_array("system:index").getInfo()
    if max_scenes:
        ids = ids[:max_scenes]
    print(f"{len(ids)} dry-season scenes")

    kernel = ee.Kernel.square(radius=patch // 2, units="pixels")
    imgs, scal, ys, lons, lats, dates = [], [], [], [], [], []

    for i, sid in enumerate(ids, 1):
        img = ee.Image(col.filter(ee.Filter.eq("system:index", sid)).first())
        date = img.date().format("YYYY-MM-dd").getInfo()

        # Spatial bands get the full neighbourhood; LST is needed only at the centre.
        nbr = img.select(SPATIAL_CHANNELS).neighborhoodToArray(kernel)
        stack = nbr.addBands(img.select(["lst"] + [c for c in SCALAR_CHANNELS
                                                   if c not in ("lon", "lat")]))

        try:
            fc = stack.sample(region=aoi, scale=SCALE, numPixels=per_scene,
                              seed=seed + i, dropNulls=True, geometries=True).getInfo()
        except Exception as exc:
            print(f"  [{i}/{len(ids)}] {date}: sample failed ({exc}); skipping")
            continue

        kept = 0
        for f in fc.get("features", []):
            p = f["properties"]
            try:
                arrs = [np.asarray(p[c], dtype="float32") for c in SPATIAL_CHANNELS]
            except (KeyError, TypeError, ValueError):
                continue
            if any(a.shape != (patch, patch) for a in arrs) or p.get("lst") is None:
                continue
            if any(not np.isfinite(a).all() for a in arrs):
                continue
            lon, lat = f["geometry"]["coordinates"][:2]
            sv = [lon if c == "lon" else lat if c == "lat" else p.get(c)
                  for c in SCALAR_CHANNELS]
            if any(v is None for v in sv):
                continue
            imgs.append(np.stack(arrs))
            scal.append(np.asarray(sv, dtype="float32"))
            ys.append(float(p["lst"]))
            lons.append(lon); lats.append(lat); dates.append(date)
            kept += 1
        print(f"  [{i}/{len(ids)}] {date}: {kept} chips")

    if not imgs:
        raise SystemExit("no chips exported — check the AOI, date window and cloud masking")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = out or f"{OUT_DIR}/kochi_chips_p{patch}.npz"
    np.savez_compressed(
        out,
        img=np.stack(imgs).astype("float32"),
        scalar=np.stack(scal).astype("float32"),
        y=np.asarray(ys, dtype="float32"),
        lon=np.asarray(lons, dtype="float32"),
        lat=np.asarray(lats, dtype="float32"),
        date=np.asarray(dates),
        channels=np.asarray(SPATIAL_CHANNELS),
        scalars=np.asarray(SCALAR_CHANNELS),
        patch=np.asarray(patch),
    )
    print(f"\nWrote {out}: {len(imgs):,} dense {patch}x{patch} chips, "
          f"{len(SPATIAL_CHANNELS)} spatial channels")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--per-scene", type=int, default=1200)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    export(a.config, a.patch, a.per_scene, a.max_scenes, a.seed, a.out)
