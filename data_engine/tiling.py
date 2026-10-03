"""Web-Mercator map tiles from a raster, with no Earth Engine in sight.

Used to BAKE the thermal overlay: the median land-surface-temperature composite is downloaded
from Earth Engine once, cut into a standard {z}/{x}/{y}.png pyramid here, and then served as
plain static files. numpy and Pillow only, so every function can be tested offline.

GEOREFERENCING
    The raster is described by a regular grid in (longitude, Mercator y): pixel CENTRES sit at
    lon0 + col * dlon and my0 - row * dmy. That is exactly how an Earth Engine download in
    EPSG:3857 is laid out, and it makes tile sampling separable (a tile column depends only on
    longitude, a tile row only on Mercator y), which keeps it fast and easy to get right.

    The grid is not taken on trust. Raster.from_lonlat() RECOVERS it from the per-pixel
    longitude/latitude bands Earth Engine supplies and refuses to continue if the pixels do not
    sit on a regular grid, because a silent half-pixel or flipped-axis error would shift the whole
    overlay against the basemap and look plausible.

RESAMPLING
    Where a tile pixel is SMALLER than a source pixel (high zoom) it takes the nearest source
    pixel, so the 30 m blocks stay crisp and equal-sized. Where a tile pixel is LARGER (low zoom)
    it averages the TEMPERATURES under it before colouring, never the colours: averaging blue
    and red gives purple, which the colour ramp never produced and which means nothing.
"""

import math
import os

import numpy as np
from PIL import Image

TILE = 256
METRES_PER_PX_Z0 = 156543.03392804097        # at the equator, for 256 px tiles


# ───────────────────────── Mercator helpers ─────────────────────────

def mercator_y(lat_deg):
    """Mercator y in radians (+/- pi at the poles of the square world), for latitude in degrees."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    return np.log(np.tan(np.pi / 4 + lat / 2))


def lat_from_mercator_y(my):
    return np.degrees(2 * np.arctan(np.exp(np.asarray(my, dtype=np.float64))) - np.pi / 2)


def lonlat_to_tile(lon, lat, z):
    """The (x, y) tile containing a point at zoom z, clamped to the world."""
    n = 2 ** z
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - float(mercator_y(lat)) / math.pi) / 2.0 * n))
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_range(bounds, z, pad=1e-4):
    """Inclusive (x0, x1, y0, y1) of every tile touching [west, south, east, north].

    Padded by ~10 m so a tile that merely TOUCHES the edge is baked too. Mapbox decides which tiles
    to request from the source bounds, and a requested tile that was never written is a 404.
    """
    w, s, e, n = bounds
    x0, y1 = lonlat_to_tile(w - pad, s - pad, z)
    x1, y0 = lonlat_to_tile(e + pad, n + pad, z)
    return x0, x1, y0, y1


def ground_resolution_m(z, lat_deg):
    """Metres covered by one tile pixel at zoom z and a given latitude."""
    return METRES_PER_PX_Z0 * math.cos(math.radians(lat_deg)) / 2 ** z


# ───────────────────────── the raster ─────────────────────────

class Raster:
    """values[row, col] on a regular (longitude, Mercator-y) grid; NaN means no data."""

    def __init__(self, values, lon0, dlon, my0, dmy):
        self.values = np.asarray(values, dtype=np.float32)
        self.lon0, self.dlon, self.my0, self.dmy = float(lon0), float(dlon), float(my0), float(dmy)
        self.height, self.width = self.values.shape

    @classmethod
    def from_lonlat(cls, values, lon, lat, tol_px=0.25):
        """Recover the grid from per-pixel longitude and latitude arrays.

        Earth Engine returns these as float32, which is good to about 1 m: noisy next to a 30 m
        pixel but not next to a fit through hundreds of them, so the grid is a least-squares line
        through the column means and the row means. The residual of EVERY pixel against that line
        is then checked, which is what makes a rotated, flipped or irregular grid fail loudly.
        """
        values = np.asarray(values)
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        if not (values.shape == lon.shape == lat.shape) or values.ndim != 2:
            raise ValueError(f"values/lon/lat must be the same 2-D shape, got "
                             f"{values.shape}/{lon.shape}/{lat.shape}")
        h, w = values.shape
        cols, rows = np.arange(w), np.arange(h)
        my = mercator_y(lat)
        dlon, lon0 = np.polyfit(cols, lon.mean(axis=0), 1)
        slope, my0 = np.polyfit(rows, my.mean(axis=1), 1)
        dmy = -slope
        if dlon <= 0 or dmy <= 0:
            raise ValueError("raster is flipped: longitude must increase with column and "
                             "latitude must decrease with row")
        res_lon = np.abs(lon - (lon0 + dlon * cols[None, :])).max() / dlon
        res_my = np.abs(my - (my0 - dmy * rows[:, None])).max() / dmy
        if max(res_lon, res_my) > tol_px:
            raise ValueError(f"pixels do not sit on a regular (lon, Mercator-y) grid: worst "
                             f"residual {max(res_lon, res_my):.2f} px (limit {tol_px})")
        return cls(values, lon0, dlon, my0, dmy)

    @property
    def bounds(self):
        """[west, south, east, north] of the pixel EDGES (half a pixel beyond the outer centres)."""
        w = self.lon0 - self.dlon / 2
        e = self.lon0 + (self.width - 1) * self.dlon + self.dlon / 2
        n = float(lat_from_mercator_y(self.my0 + self.dmy / 2))
        s = float(lat_from_mercator_y(self.my0 - (self.height - 0.5) * self.dmy))
        return [w, s, e, n]

    @property
    def native_metres(self):
        """Ground size of one source pixel at the raster's centre latitude."""
        lat_mid = float(lat_from_mercator_y(self.my0 - self.dmy * self.height / 2))
        return self.dlon * (math.pi / 180.0) * 6378137.0 * math.cos(math.radians(lat_mid))

    def sample(self, lon, lat):
        """Nearest-pixel value at points (arrays of lon, lat); NaN outside the raster."""
        lon = np.asarray(lon, dtype=np.float64)
        c = np.rint((lon - self.lon0) / self.dlon).astype(np.int64)
        r = np.rint((self.my0 - mercator_y(lat)) / self.dmy).astype(np.int64)
        ok = (c >= 0) & (c < self.width) & (r >= 0) & (r < self.height)
        out = np.full(c.shape, np.nan, dtype=np.float32)
        out[ok] = self.values[r[ok], c[ok]]
        return out


# ───────────────────────── tiles ─────────────────────────

def tile_values(raster, z, x, y, ss=1):
    """The 256x256 temperatures of one tile. ss > 1 averages an ss x ss grid under each pixel."""
    n = TILE * ss
    world = TILE * 2 ** z
    px = x * TILE + (np.arange(n) + 0.5) / ss                 # world pixel coords of sub-centres
    py = y * TILE + (np.arange(n) + 0.5) / ss
    lon = px / world * 360.0 - 180.0
    my = np.pi * (1.0 - 2.0 * py / world)
    c = np.rint((lon - raster.lon0) / raster.dlon).astype(np.int64)
    r = np.rint((raster.my0 - my) / raster.dmy).astype(np.int64)
    cv = (c >= 0) & (c < raster.width)
    rv = (r >= 0) & (r < raster.height)

    sub = np.full((n, n), np.nan, dtype=np.float32)
    if cv.any() and rv.any():
        sub[np.ix_(rv, cv)] = raster.values[np.ix_(r[rv], c[cv])]
    if ss == 1:
        return sub

    blocks = sub.reshape(TILE, ss, TILE, ss)
    valid = np.isfinite(blocks)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean.astype(np.float32)


def parse_palette(palette):
    """['313695', '#4575b4', ...] -> float array of RGB stops."""
    return np.array([[int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)] for h in palette],
                    dtype=np.float64)


def colourise(values, vmin, vmax, palette):
    """RGBA uint8 for an array of temperatures. NaN is fully transparent.

    Matches how Earth Engine paints a visualised image: the palette stops are spread evenly over
    [vmin, vmax], colours interpolate linearly between them, and values beyond either end clamp to
    the end colour instead of disappearing.
    """
    stops = parse_palette(palette)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    t = np.clip((np.where(valid, values, vmin) - vmin) / (vmax - vmin), 0.0, 1.0)
    pos = t * (len(stops) - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int64), len(stops) - 2)
    f = (pos - i0)[..., None]
    rgb = stops[i0] * (1.0 - f) + stops[i0 + 1] * f
    alpha = np.where(valid, 255, 0)
    return np.concatenate([np.rint(rgb), alpha[..., None]], axis=-1).astype(np.uint8)


def write_png(rgba, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path, format="PNG", optimize=True)
    return os.path.getsize(path)


def supersample_for(z, lat_deg, native_m, cap=8):
    """How many sub-samples per tile-pixel side are needed to average properly at this zoom."""
    res = ground_resolution_m(z, lat_deg)
    return int(min(cap, max(1, math.ceil(res / native_m))))


def bake_pyramid(raster, bounds, out_dir, vmin, vmax, palette, zmin, zmax, log=print):
    """Write {out_dir}/{z}/{x}/{y}.png for every tile touching `bounds`, zmin..zmax inclusive.

    EVERY tile in the range is written, fully transparent ones included (a few dozen bytes each).
    The Mapbox raster source is told the bounds, so it asks for exactly this set and a missing file
    would be a hard 404 rather than an empty tile.
    """
    lat_mid = (bounds[1] + bounds[3]) / 2
    stats = {"tiles": 0, "bytes": 0, "empty": 0, "per_zoom": {}}
    for z in range(zmin, zmax + 1):
        ss = supersample_for(z, lat_mid, raster.native_metres)
        x0, x1, y0, y1 = tile_range(bounds, z)
        count = size = 0
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                vals = tile_values(raster, z, x, y, ss)
                if not np.isfinite(vals).any():
                    stats["empty"] += 1
                size += write_png(colourise(vals, vmin, vmax, palette),
                                  os.path.join(out_dir, str(z), str(x), f"{y}.png"))
                count += 1
        stats["tiles"] += count
        stats["bytes"] += size
        stats["per_zoom"][z] = {"tiles": count, "bytes": size, "supersample": ss,
                                "tile_px_m": round(ground_resolution_m(z, lat_mid), 1)}
        log(f"  z{z:<2} {count:>4} tiles  {size / 1024:>8.1f} KB   "
            f"({ground_resolution_m(z, lat_mid):6.1f} m/px, {ss}x{ss} averaging)")
    return stats
