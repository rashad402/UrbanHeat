"""Synthetic stand-in for the Kochi dataset — lets the tests run without Earth Engine.

The real table (data/processed/kochi_samples.parquet) is built from Landsat/ERA5 via GEE and is
not in the repository. These fixtures generate a table with the SAME CONTRACT and, critically,
the same two properties the code under test depends on:

  1. A NARROW ALBEDO RANGE. Built surfaces sit near 0.15, vegetation near 0.10, so nothing in
     the sample approaches a cool roof's 0.50. That is the data gap models/synthetic.py exists
     to fill.
  2. THE LAND-COVER CONFOUND. Built pixels are both brighter AND hotter than vegetated ones, so
     albedo correlates POSITIVELY with LST in-sample even though the energy balance says
     brightening a surface cools it. A data-only model learns the correlation and gets the
     counterfactual sign backwards — the failure the physics term exists to prevent.

Labels come from the SEB equilibrium plus a land-cover-dependent offset and noise, so the table
is physically coherent rather than arbitrary.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import sebal as S  # noqa: E402

KOCHI_LON, KOCHI_LAT = 76.28, 9.97


def synthetic_dataset(n=6000, n_wards=12, n_dates=6, seed=7, noise_k=0.8):
    """Return a DataFrame matching the dataset contract of data_engine/build_dataset.py."""
    rng = np.random.default_rng(seed)

    # Land cover: a built fraction per pixel drives NDVI, NDBI and albedo together.
    built = rng.beta(2.2, 1.6, size=n)
    ndvi = np.clip(0.72 - 0.60 * built + rng.normal(0, 0.05, n), -0.05, 0.9)
    ndbi = np.clip(-0.35 + 0.65 * built + rng.normal(0, 0.05, n), -0.9, 0.9)
    # Concrete is brighter than canopy, but the whole range stays far below a cool roof.
    albedo = np.clip(0.095 + 0.055 * built + rng.normal(0, 0.012, n), 0.04, 0.26)

    # Spatial layout: a grid wide enough that spatial-block splitting has blocks to work with.
    side = int(np.ceil(np.sqrt(n)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    lon = KOCHI_LON + (gx.ravel()[:n] / side) * 0.13
    lat = KOCHI_LAT + (gy.ravel()[:n] / side) * 0.13

    # Dates spread across the dry season (Dec-Feb) and pre-monsoon (Mar-Apr), so the season
    # filter in api/planner.py has both buckets to select.
    months = np.array([12, 1, 2, 3, 4, 1])[:n_dates]
    years = 2016 + np.arange(n_dates) % 8
    didx = rng.integers(0, n_dates, size=n)
    date = np.array([f"{years[i]}-{months[i]:02d}-1{(i % 9)}" for i in didx])

    # Per-date climate, as ERA5-Land would supply it.
    s_down_d = rng.uniform(580, 720, n_dates)
    t_air_d = rng.uniform(299.0, 304.0, n_dates)
    rh_d = rng.uniform(55, 80, n_dates)
    wind_d = rng.uniform(0.8, 3.0, n_dates)
    s_down, t_air = s_down_d[didx], t_air_d[didx]
    rh, wind = rh_d[didx], wind_d[didx]

    # Physically coherent target: SEB equilibrium, plus the radiometric offset that makes built
    # surfaces read hotter than the aerodynamic closure predicts, plus observation noise.
    t_eq = S.equilibrium_temperature(ndvi, albedo, s_down, t_air, rh, wind)
    lst = t_eq + 2.5 * built + rng.normal(0, noise_k, n)
    lst = np.clip(lst, 290.0, 330.0)

    ward_id = np.array([f"W{(i % n_wards) + 1:02d}" for i in range(n)])

    return pd.DataFrame({
        "lon": lon, "lat": lat, "date": date,
        "ndvi": ndvi, "ndbi": ndbi, "albedo": albedo,
        "s_down": s_down, "t_air": t_air, "rh": rh, "wind": wind,
        "lst": lst, "lst_source": "synthetic", "ward_id": ward_id,
    })


def synthetic_wards(df, path):
    """Write a ward GeoJSON whose polygons tile the pixels, including some with NO data.

    The no-data wards matter: api/planner.py must mark them has_data=false and the UI must
    render them as deliberately inert rather than broken.
    """
    import json

    ids = sorted(df.ward_id.unique())
    feats = []
    for i, wid in enumerate(ids + ["W90", "W91"]):      # two wards that own no pixels
        sub = df[df.ward_id == wid]
        if len(sub):
            w, e = float(sub.lon.min()), float(sub.lon.max())
            s, n = float(sub.lat.min()), float(sub.lat.max())
            # Pixels are striped across wards, so a bbox would overlap every other ward.
            # A small box around the centroid keeps the polygons disjoint.
            cx, cy = (w + e) / 2, (s + n) / 2
            w, e, s, n = cx - 0.004, cx + 0.004, cy - 0.004, cy + 0.004
        else:
            w, e = KOCHI_LON + 0.20 + i * 0.01, KOCHI_LON + 0.205 + i * 0.01
            s, n = KOCHI_LAT + 0.20, KOCHI_LAT + 0.205
        feats.append({
            "type": "Feature",
            "properties": {"ward_id": wid, "ward_name": f"Ward {wid}", "ward_no": i + 1,
                           "area_sqkm": round(0.6 + 0.1 * (i % 5), 3)},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
        })
    gj = {"type": "FeatureCollection", "features": feats}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gj, fh)
    return gj


def write_fixture(dirpath, n=6000, seed=7):
    """Materialise a parquet + ward GeoJSON pair and return their paths."""
    os.makedirs(dirpath, exist_ok=True)
    df = synthetic_dataset(n=n, seed=seed)
    pq = os.path.join(dirpath, "samples.parquet")
    gj = os.path.join(dirpath, "wards.geojson")
    df.to_parquet(pq)
    synthetic_wards(df, gj)
    return pq, gj
