"""Tests for the baked-overlay tile maths (data_engine/tiling.py).

Run:  python tests/test_tiling.py

Offline: numpy and Pillow only. The failure these guard against is not a crash but a plausible
looking overlay in the WRONG PLACE: shifted half a pixel, mirrored north-south, or colours averaged
instead of temperatures. Each of those draws a believable heat map, so nothing downstream would
ever flag it, and a planner would read the wrong street as the hot one. So most of what follows
checks WHERE a known pixel lands, against an independent computation of the slippy-map formula.
"""

import math
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_engine import tiling as T  # noqa: E402

PASS, FAIL = [], []
R_EARTH = 6378137.0
PALETTE = ["313695", "4575b4", "74add1", "abd9e9", "fee090", "fdae61", "f46d43", "d73027", "a50026"]


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def make_raster(values, lon0=76.20, lat_top=10.05, scale_m=30.0):
    """A raster on the same grid an Earth Engine EPSG:3857 download at `scale_m` has."""
    step = scale_m / R_EARTH                                      # radians, in lon and Mercator y
    return T.Raster(values, lon0, math.degrees(step), float(T.mercator_y(lat_top)), step)


def lonlat_arrays(r):
    """Per-pixel lon/lat bands as Earth Engine would return them (float32)."""
    cols, rows = np.arange(r.width), np.arange(r.height)
    lon = np.tile(r.lon0 + r.dlon * cols, (r.height, 1))
    lat = np.tile(T.lat_from_mercator_y(r.my0 - r.dmy * rows)[:, None], (1, r.width))
    return lon.astype(np.float32), lat.astype(np.float32)


def slippy_pixel(lon, lat, z):
    """INDEPENDENT reference: world pixel of a point, from the textbook slippy-map formulae."""
    n = 256 * 2 ** z
    px = (lon + 180.0) / 360.0 * n
    py = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return px, py


def main():
    print("Overlay tiling tests\n" + "=" * 60)

    # ───────────── tile addressing ─────────────
    check("San Francisco is tile 163/395 at z10 (a well-known reference)",
          T.lonlat_to_tile(-122.4194, 37.7749, 10) == (163, 395),
          str(T.lonlat_to_tile(-122.4194, 37.7749, 10)))
    check("tile numbering matches the textbook formula for Kochi at z8-15",
          all(T.lonlat_to_tile(76.27, 9.97, z) ==
              (int(slippy_pixel(76.27, 9.97, z)[0] // 256), int(slippy_pixel(76.27, 9.97, z)[1] // 256))
              for z in range(8, 16)))
    lat = np.array([-60.0, -10.0, 0.0, 9.97, 45.0, 80.0])
    check("Mercator y round-trips through latitude",
          np.allclose(T.lat_from_mercator_y(T.mercator_y(lat)), lat, atol=1e-9))
    check("a 30 m pixel is ~29.5 m on the ground at Kochi's latitude",
          abs(make_raster(np.zeros((40, 40))).native_metres - 30 * math.cos(math.radians(10))) < 0.3,
          f"{make_raster(np.zeros((40, 40))).native_metres:.2f} m")

    # ───────────── recovering the grid ─────────────
    vals = np.random.default_rng(1).normal(37, 2, (60, 80)).astype(np.float32)
    truth = make_raster(vals)
    lon, lat = lonlat_arrays(truth)
    rec = T.Raster.from_lonlat(vals, lon, lat)
    check("grid recovered from float32 lon/lat bands to a tiny fraction of a pixel",
          abs(rec.lon0 - truth.lon0) < 0.02 * truth.dlon and abs(rec.dlon / truth.dlon - 1) < 1e-3
          and abs(rec.my0 - truth.my0) < 0.02 * truth.dmy and abs(rec.dmy / truth.dmy - 1) < 1e-3,
          f"lon0 off {abs(rec.lon0 - truth.lon0) / truth.dlon:.4f} px, my0 off "
          f"{abs(rec.my0 - truth.my0) / truth.dmy:.4f} px")
    for label, args in (
            ("a north-south flipped raster", (vals, lon, lat[::-1])),
            ("an east-west flipped raster", (vals, lon[:, ::-1], lat)),
            ("an irregular grid", (vals, lon + np.random.default_rng(2).normal(0, 2e-4, lon.shape)
                                   .astype(np.float32), lat))):
        try:
            T.Raster.from_lonlat(*args)
            check(f"{label} is refused", False)
        except ValueError as exc:
            check(f"{label} is refused", True, str(exc)[:60])

    # ───────────── a known pixel lands where the textbook says ─────────────
    z = 15
    mark_lon, mark_lat = 76.2650, 9.9700
    vals = np.full((400, 400), 20.0, dtype=np.float32)
    r0 = make_raster(vals, lon0=76.2400, lat_top=9.9900)
    c = int(round((mark_lon - r0.lon0) / r0.dlon))
    rr = int(round((r0.my0 - float(T.mercator_y(mark_lat))) / r0.dmy))
    r0.values[rr, c] = 99.0
    tx, ty = T.lonlat_to_tile(mark_lon, mark_lat, z)
    tile = T.tile_values(r0, z, tx, ty, ss=1)
    hit = np.argwhere(tile == 99.0)
    px, py = slippy_pixel(r0.lon0 + c * r0.dlon, float(T.lat_from_mercator_y(r0.my0 - rr * r0.dmy)), z)
    exp_col, exp_row = px - tx * 256, py - ty * 256            # centre of the marked pixel, in-tile
    check("the marked pixel appears in the tile, as a block of ~6 px (30 m at z15)",
          len(hit) > 0 and 4 <= (hit[:, 0].max() - hit[:, 0].min() + 1) <= 8,
          f"{len(hit)} px")
    check("... and is centred where the independent slippy-map formula puts it",
          len(hit) > 0 and abs(hit[:, 1].mean() + 0.5 - exp_col) < 1.0
          and abs(hit[:, 0].mean() + 0.5 - exp_row) < 1.0,
          f"got ({hit[:, 1].mean() + 0.5:.1f}, {hit[:, 0].mean() + 0.5:.1f}) "
          f"expected ({exp_col:.1f}, {exp_row:.1f})")

    # north must be at the TOP of the tile: temperature rising northward must read top > bottom
    ramp = np.tile(np.linspace(40, 30, 400, dtype=np.float32)[:, None], (1, 400))   # row 0 = north
    rr2 = make_raster(ramp, lon0=76.2400, lat_top=9.9900)
    tx, ty = T.lonlat_to_tile(76.2500, 9.9800, 14)
    t2 = T.tile_values(rr2, 14, tx, ty)
    check("north is at the top of the tile (no north-south mirroring)",
          np.nanmean(t2[:60]) > np.nanmean(t2[-60:]),
          f"top {np.nanmean(t2[:60]):.1f} vs bottom {np.nanmean(t2[-60:]):.1f}")
    ew = np.tile(np.linspace(30, 40, 400, dtype=np.float32)[None, :], (400, 1))     # col 0 = west
    t3 = T.tile_values(make_raster(ew, lon0=76.2400, lat_top=9.9900), 14, tx, ty)
    check("east is at the right of the tile (no east-west mirroring)",
          np.nanmean(t3[:, -60:]) > np.nanmean(t3[:, :60]))

    # ───────────── pixels outside the raster ─────────────
    far = T.tile_values(rr2, 14, tx + 20, ty + 20)
    check("a tile entirely outside the raster is all NaN", not np.isfinite(far).any())
    check("a tile straddling the raster edge is partly data, partly NaN",
          0 < np.isfinite(T.tile_values(rr2, 12, *T.lonlat_to_tile(76.2400, 9.9900, 12))).mean() < 1)

    # ───────────── averaging temperatures, not colours ─────────────
    checker = np.indices((400, 400)).sum(axis=0) % 2 * 20.0 + 10.0        # 10 and 30, alternating
    rc = make_raster(checker.astype(np.float32), lon0=76.2400, lat_top=9.9900)
    tx, ty = T.lonlat_to_tile(76.2500, 9.9800, 10)
    mean_tile = T.tile_values(rc, 10, tx, ty, ss=T.supersample_for(10, 9.97, rc.native_metres))
    inside = mean_tile[np.isfinite(mean_tile)]
    check("a low-zoom tile averages the TEMPERATURES under each pixel (10 and 30 -> 20)",
          inside.size > 0 and abs(float(np.median(inside)) - 20.0) < 1.5,
          f"median {float(np.median(inside)):.2f}")
    check("supersampling is 1 where the tile pixel is finer than the data, larger where coarser",
          T.supersample_for(15, 9.97, 29.5) == 1 and T.supersample_for(13, 9.97, 29.5) == 1
          and T.supersample_for(8, 9.97, 29.5) > 4)

    # ───────────── colour ramp ─────────────
    pal = T.parse_palette(PALETTE)
    cols = T.colourise(np.array([31.0, 41.0, 36.0, 20.0, 60.0, np.nan]), 31, 41, PALETTE)
    check("the minimum maps to the first palette colour and the maximum to the last",
          tuple(cols[0, :3]) == tuple(pal[0].astype(int)) and tuple(cols[1, :3]) == tuple(pal[-1].astype(int)))
    check("the midpoint is the middle palette stop", tuple(cols[2, :3]) == tuple(pal[4].astype(int)),
          str(tuple(cols[2, :3])))
    check("values beyond the range clamp to the end colours instead of vanishing",
          tuple(cols[3, :3]) == tuple(cols[0, :3]) and cols[3, 3] == 255
          and tuple(cols[4, :3]) == tuple(cols[1, :3]) and cols[4, 3] == 255)
    check("NaN (cloud, water, outside the area) is fully transparent", cols[5, 3] == 0)
    ramp_rgb = T.colourise(np.linspace(31, 41, 200), 31, 41, PALETTE)[:, :3].astype(int)
    check("the ramp changes smoothly: no adjacent step jumps by more than ~25 per channel",
          np.abs(np.diff(ramp_rgb, axis=0)).max() <= 25)

    # ───────────── writing the pyramid ─────────────
    tmp = tempfile.mkdtemp(prefix="urbanheat-tiles-")
    small = make_raster(np.random.default_rng(3).uniform(33, 40, (300, 300)).astype(np.float32),
                        lon0=76.2400, lat_top=9.9900)
    small.values[:40] = np.nan                                      # a clipped-away strip
    b = small.bounds
    stats = T.bake_pyramid(small, b, tmp, 31, 41, PALETTE, 9, 13, log=lambda *_: None)
    expect = sum((x1 - x0 + 1) * (y1 - y0 + 1)
                 for x0, x1, y0, y1 in (T.tile_range(b, z) for z in range(9, 14)))
    files = [os.path.join(d, f) for d, _, fs in os.walk(tmp) for f in fs]
    check("every tile in range is written, none missing", len(files) == expect == stats["tiles"],
          f"{len(files)} files, expected {expect}")
    check("files sit at {z}/{x}/{y}.png", all(f.endswith(".png") for f in files)
          and os.path.exists(os.path.join(tmp, "13", str(T.tile_range(b, 13)[0]),
                                          f"{T.tile_range(b, 13)[2]}.png")))
    im = Image.open(files[0])
    check("tiles are 256x256 RGBA", im.size == (256, 256) and im.mode == "RGBA")
    top_tile = Image.open(os.path.join(tmp, "13", str(T.tile_range(b, 13)[0]), f"{T.tile_range(b, 13)[2]}.png"))
    a = np.asarray(top_tile)[..., 3]
    check("masked pixels are transparent and valid ones opaque in the same tile",
          (a == 0).any() and (a == 255).any())

    print("=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
