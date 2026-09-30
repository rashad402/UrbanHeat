"""Google Earth Engine extraction (plan §5.1).

Usage:
    python data_engine/gee_export.py --config configs/data_config.yaml

Extracts, for the AOI and time window in the config:
  - Landsat 8/9 C2 L2 surface temperature (ST_B10) + QA_PIXEL
  - Sentinel-2 SR bands for NDVI/NDBI/albedo + SCL
  - MODIS MOD11A2 8-day LST (gap-fill source)
  - ERA5-Land climate variables (t_air, dewpoint->RH, u10/v10->wind, s_down)

Exports to Google Drive / GCS as GeoTIFFs (tiled if large), then downloaded locally.
Nothing here should require manual clicking in the GEE Code Editor.
"""

import argparse


def init_ee():
    """Authenticate + initialize Earth Engine (expects `earthengine authenticate` done)."""
    raise NotImplementedError


def export_all(config):
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    args = ap.parse_args()
    export_all(args.config)
