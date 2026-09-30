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


def ee_climate_bands(overpass_millis):
    """Return an ee.Image with bands [t_air, rh, wind, s_down] at the overpass time."""
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

    return t_air.addBands([rh, wind, s_down])
