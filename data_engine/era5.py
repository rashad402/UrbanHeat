"""ERA5-Land climate ingestion (plan §5.1).

Attaches climate forcing to each Landsat scene at its overpass time:
  - t_air  [K]      temperature_2m
  - rh     [%]      from temperature_2m + dewpoint_temperature_2m (Magnus formula)
  - wind   [m/s]    sqrt(u10^2 + v10^2)
  - s_down [W/m^2]  instantaneous surface solar radiation, from the hourly accumulation

ERA5-Land is ~11 km native resolution, so over a ~15 km city these are near-constant
regional forcings — that is expected and fine (climate drives the whole scene).

The numpy helpers (relative_humidity, wind_speed) are for local/array use; the ee_* helpers
build Earth Engine images used server-side by build_dataset.py.
"""

import ee

ERA5_HOURLY = "ECMWF/ERA5_LAND/HOURLY"


# ---- numpy / scalar helpers (local use) ----

def relative_humidity(t_air_k, dewpoint_k):
    """RH [%] from air temperature and dewpoint (Magnus/Tetens), inputs in Kelvin."""
    import math
    def es(tk):
        tc = tk - 273.15
        return 6.112 * math.exp(17.67 * tc / (tc + 243.5))
    return max(0.0, min(100.0, 100.0 * es(dewpoint_k) / es(t_air_k)))


def wind_speed(u10, v10):
    return (u10 ** 2 + v10 ** 2) ** 0.5


# ---- Earth Engine helpers (server-side) ----

def _es_ee(t_kelvin_img):
    """Saturation vapour pressure [hPa] via Magnus, on an ee.Image of temperature in K."""
    tc = t_kelvin_img.subtract(273.15)
    return tc.multiply(17.67).divide(tc.add(243.5)).exp().multiply(6.112)


def ee_solar_radiation_wm2(overpass_millis):
    """Instantaneous downwelling solar radiation [W/m^2] at the overpass hour.

    ERA5-Land 'surface_solar_radiation_downwards' is accumulated from 00 UTC, so the
    instantaneous flux is the hour-over-hour difference divided by 3600 s.
    """
    t = ee.Date(overpass_millis)
    hour_start = ee.Date.fromYMD(t.get("year"), t.get("month"), t.get("day")).advance(t.get("hour"), "hour")
    this_h = ee.ImageCollection(ERA5_HOURLY).filterDate(hour_start, hour_start.advance(1, "hour")).first()
    prev_h = ee.ImageCollection(ERA5_HOURLY).filterDate(hour_start.advance(-1, "hour"), hour_start).first()

    ssrd_now = ee.Image(this_h).select("surface_solar_radiation_downwards")
    # If it's the 00:00 hour there is no previous accumulation to subtract.
    is_midnight = ee.Number(t.get("hour")).eq(0)
    ssrd_prev = ee.Image(ee.Algorithms.If(
        is_midnight,
        ee.Image.constant(0).rename("surface_solar_radiation_downwards"),
        ee.Image(prev_h).select("surface_solar_radiation_downwards"),
    ))
    return ssrd_now.subtract(ssrd_prev).divide(3600).rename("s_down").max(0)


# ---- coastal gap fill ----
#
# ERA5-Land carries its own coarse land-sea mask on the ~11 km grid, and over Kochi that mask
# drops the cells covering the south and west of the corporation — Fort Kochi, Mattancherry,
# Thevara, Palluruthy, Konthuruthy. Every climate band is null there even though the Landsat
# scene has tens of thousands of valid land pixels in the same place. build_dataset samples
# with dropNulls=True, so those pixels were silently discarded and the training table stopped
# dead at lon 76.25 / lat 9.95 — the exact 0.1 deg cell edges. That cost ~20% of the
# corporation and left 27 of 77 wards with no data at all.
#
# So the bands are extended across those cells from their nearest valid neighbours. At 11 km
# ERA5-Land cannot resolve intra-city variation in the first place — the module docstring
# above already calls these near-constant regional forcings — so what filling invents is small
# beside what dropping a fifth of the city destroyed. Pixels that rely on it are flagged
# (climate_filled) so the choice stays visible downstream and in the report.

FILL_RADIUS_CELLS = 3      # ~33 km; the coastal gap is 1-3 cells wide
FILL_PASSES = 2


def _fill_masked(img, proj, radius_cells=FILL_RADIUS_CELLS, passes=FILL_PASSES):
    """Extend `img` over cells its own mask drops, using the mean of valid neighbours.

    Two details matter and both are easy to get wrong:

    1. The kernel is sized in PIXELS, not metres, and the computation is pinned to ERA5's
       native projection with reproject(). A kernel given in metres is re-expressed at
       whatever scale the caller requests — build_dataset samples at 30 m, where a 33 km
       radius becomes a ~1100-pixel kernel that Earth Engine refuses outright.
    2. skipMasked=False is what makes the reducer run AT a masked pixel using its valid
       neighbours. With the default (True) the output stays masked and nothing is filled.
    """
    names = img.bandNames()
    kernel = ee.Kernel.circle(radius=radius_cells, units="pixels")
    out = img.reproject(proj)
    for _ in range(passes):
        neighbours = out.reduceNeighborhood(
            reducer=ee.Reducer.mean(), kernel=kernel, skipMasked=False,
        ).rename(names).reproject(proj)
        out = out.unmask(neighbours).reproject(proj)
    return out


def ee_climate_bands(overpass_millis, fill_gaps=True):
    """Return an ee.Image with bands [t_air, rh, wind, s_down] at the overpass time.

    With fill_gaps (the default) the bands are extended over ERA5-Land's coastal land-sea
    mask gaps and a 0/1 `climate_filled` band says which pixels needed it. Pass False to see
    the raw product — that is what the first version of the training table was built on.
    """
    t = ee.Date(overpass_millis)
    hour_start = ee.Date.fromYMD(t.get("year"), t.get("month"), t.get("day")).advance(t.get("hour"), "hour")
    era = ee.Image(ee.ImageCollection(ERA5_HOURLY)
                   .filterDate(hour_start, hour_start.advance(1, "hour")).first())

    t_air = era.select("temperature_2m").rename("t_air")               # K
    dew = era.select("dewpoint_temperature_2m")
    u = era.select("u_component_of_wind_10m")
    v = era.select("v_component_of_wind_10m")
    wind = u.hypot(v).rename("wind")                                    # m/s
    rh = _es_ee(dew).divide(_es_ee(t_air)).multiply(100).clamp(0, 100).rename("rh")
    s_down = ee_solar_radiation_wm2(overpass_millis)                   # W/m^2

    bands = t_air.addBands([rh, wind, s_down])
    if not fill_gaps:
        return bands

    # The flag comes from the ORIGINAL mask, before anything is filled.
    was_missing = t_air.mask().Not().unmask(1).rename("climate_filled")
    proj = era.select("temperature_2m").projection()
    return _fill_masked(bands, proj).addBands(was_missing)
