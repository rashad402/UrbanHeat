"""ERA5-Land climate ingestion (plan §5.1).

Resample hourly ERA5-Land to satellite overpass time and derive:
  - t_air  from temperature_2m [K]
  - rh     from temperature_2m + dewpoint_temperature_2m (Magnus formula)
  - wind   = sqrt(u10^2 + v10^2)
  - s_down from surface_solar_radiation_downwards [J m^-2 -> W m^-2]
"""


def relative_humidity(t_air_k, dewpoint_k):
    """RH [%] from air temperature and dewpoint (Magnus/Tetens)."""
    raise NotImplementedError


def wind_speed(u10, v10):
    return (u10 ** 2 + v10 ** 2) ** 0.5


def load_era5_at_overpass(config, overpass_times):
    """Return a table of climate variables aligned to each satellite scene time."""
    raise NotImplementedError
