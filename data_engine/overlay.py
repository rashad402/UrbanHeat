"""The thermal overlay on the Earth Engine side: what it is, how far it extends, how it is coloured.

ONE definition, used by two callers that must never disagree:
    scripts/bake_overlay.py   computes it once and saves it as static map tiles
    api/planner.py            falls back to serving it live from Earth Engine when no baked tiles exist

If these were two copies of the same recipe they would drift, and the app would show a different
heat map depending on whether the tiles had been baked. Keeping the recipe here is what makes the
fallback honest.

WHAT THE LAYER IS
    A per-pixel MEDIAN of land surface temperature over every usable dry-season Landsat 8/9 scene,
    so each pixel shows what it is typically like, not what it was on one day. Cloud and water are
    masked in each scene before the median (see data_engine.build_dataset.process_scene).
"""

import math

import ee

from .build_dataset import aoi_geometry, build_collection

# The colour ramp. Mirrored in web/planner/intro.css (the loader), which paints before the API
# has answered; keep the two in step if it ever changes.
PALETTE = ["313695", "4575b4", "74add1", "abd9e9", "fee090",
           "fdae61", "f46d43", "d73027", "a50026"]

DEFAULT_RANGE = (31, 41)      # used only if the percentile query fails; the range is presentation

SCENE_MARGIN = 0.12           # fractional padding around the corporation, for geographic context
MIN_ASPECT = 1.0              # widen east-west so the map fills a landscape viewport


def scene_bounds(aoi, margin=SCENE_MARGIN, min_aspect=MIN_ASPECT):
    """[west, south, east, north]: the corporation's bbox, padded, widened toward landscape.

    This rectangle is the extent of the thermal layer. It is deliberately larger than the
    corporation so the layer covers the whole map on screen and a drawn area anywhere in view
    returns data. One Earth Engine round trip.
    """
    b = aoi.bounds().coordinates().getInfo()[0]
    w, s, e, n = b[0][0], b[0][1], b[2][0], b[2][1]
    dw, dh = e - w, n - s
    w -= dw * margin; e += dw * margin
    s -= dh * margin; n += dh * margin
    dw, dh = e - w, n - s
    if dw / dh < min_aspect:                      # too portrait -> grow sideways
        need = min_aspect * dh - dw
        w -= need / 2; e += need / 2
    return [w, s, e, n]


def dry_season_collection(cfg, aoi):
    """Every cloud-masked, water-masked scene in the dry-season months of the configured window."""
    dry = cfg["time"]["dry_season_months"]
    col = build_collection(cfg, aoi, cfg["time"]["start"], cfg["time"]["end"])
    return (col.map(lambda im: im.set("month", ee.Image(im).date().get("month")))
               .filter(ee.Filter.inList("month", dry)))


def build_composite(cfg, bounds=None):
    """The median composite and everything derived from it.

    `bounds` skips the one Earth Engine round trip that computes them, for callers (the planner,
    when baked tiles exist) that already know the extent from the bake manifest.
    Returns a dict: aoi, bounds, region, col, feat (all bands), lst_c (single band, degC).
    """
    aoi = aoi_geometry(cfg)
    col = dry_season_collection(cfg, aoi)
    bounds = list(bounds) if bounds else scene_bounds(aoi)
    region = ee.Geometry.Rectangle(bounds)
    feat = col.median().clip(region)
    lst_c = feat.select("lst").subtract(273.15).rename("lst_c")
    return {"aoi": aoi, "bounds": bounds, "region": region, "col": col,
            "feat": feat.addBands(lst_c), "lst_c": lst_c}


def stretch_range(lst_c, aoi, fallback=DEFAULT_RANGE):
    """The colour range, from the composite's own 5th-95th percentile over the city.

    Measured for Kochi: the middle half of all land pixels spans only ~2 degC (36.6-38.6), so any
    wide range paints most of the city one colour. 5th-95th (~31-41) lets the tails saturate and
    the genuine hot spots stand out; 2nd-98th (29-43) was too gentle to separate wards.

    Rounded outward to whole degrees so the legend reads cleanly. Any failure falls back to
    `fallback` rather than blocking startup or a bake.
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
        print(f"Could not derive the colour range ({exc}); using {fallback[0]}-{fallback[1]}")
        return fallback
