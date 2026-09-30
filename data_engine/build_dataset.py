"""Dataset assembly -> the canonical dataset artifact (plan §5.5).

Usage:
    python data_engine/build_dataset.py --config configs/data_config.yaml

Co-registers all layers to a common 30 m grid + CRS (UTM 43N), stacks features, spatially
joins ward IDs, samples to a point table, and writes:

    data/processed/kochi_samples.parquet

with columns (the DATASET INTERFACE CONTRACT — plan §3):
    x, y, lat, lon, date, ndvi, ndbi, albedo, s_down, t_air, rh, wind,
    lst_landsat, lst_source in {landsat, modis_fill}, ward_id
"""

import argparse

SCHEMA = [
    "x", "y", "lat", "lon", "date",
    "ndvi", "ndbi", "albedo", "s_down", "t_air", "rh", "wind",
    "lst_landsat", "lst_source", "ward_id",
]


def build(config_path):
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    args = ap.parse_args()
    build(args.config)
