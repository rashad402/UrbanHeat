"""Tests for the response-centred evaluation (models/response_eval.py).

Run:  python tests/test_response_eval.py

Offline: numpy and pandas only. These check that each tier measures what its docstring says it
measures, using synthetic data whose right answers are known: a constant bias must not be
penalised (it cancels in a counterfactual), a model that only knows which places are hot must
score nothing on change, and Tier B must strip out weather that moves the whole city.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import response_eval as R  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def synthetic(n_pix=400, n_dates=6, seed=0):
    """Pixels observed on several dates. LST = weather(date) + place(pixel) - 12*albedo-ish response."""
    rng = np.random.default_rng(seed)
    lon = 76.2 + rng.uniform(0, 0.1, n_pix)
    lat = 9.9 + rng.uniform(0, 0.1, n_pix)
    place = rng.normal(0, 2.0, n_pix)                      # fixed "kind of place" effect
    rows = []
    for d in range(n_dates):
        weather = rng.normal(0, 3.0)                       # moves the whole city together
        ndvi = np.clip(0.4 + rng.normal(0, 0.12, n_pix), 0, 0.9)
        albedo = np.clip(0.13 + rng.normal(0, 0.03, n_pix), 0.02, 0.9)
        lst = 310 + weather + place - 8.0 * ndvi - 20.0 * albedo
        rows.append(pd.DataFrame({"lon": lon, "lat": lat, "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=16 * d),
                                  "ndvi": ndvi, "albedo": albedo, "lst": lst, "place": place, "weather": weather}))
    return pd.concat(rows, ignore_index=True)


def main():
    df = synthetic()
    y = df["lst"].to_numpy()

    # ───────────── pairing ─────────────
    i1, i2 = R.build_pairs(df)
    pix1 = df.lon.values[i1].round(6), df.lat.values[i1].round(6)
    pix2 = df.lon.values[i2].round(6), df.lat.values[i2].round(6)
    check("pairs always join the same pixel",
          np.array_equal(pix1[0], pix2[0]) and np.array_equal(pix1[1], pix2[1]))
    check("pairs always go forward in time", bool(np.all(df.date.values[i2] > df.date.values[i1])))
    check("one consecutive pair per gap: pixels x (dates - 1)", len(i1) == 400 * 5, f"{len(i1)}")
    dup = pd.concat([df.iloc[:3], df.iloc[:3]], ignore_index=True)
    check("a pixel twice on the same date is not a pair", len(R.build_pairs(dup)[0]) == 0)

    # ───────────── what each tier rewards ─────────────
    perfect = R.tiers(df, y)
    check("a perfect model scores r = 1 and slope = 1 on every tier",
          all(abs(R.change_metrics(o, p)["r"] - 1) < 1e-9 and abs(R.change_metrics(o, p)["slope"] - 1) < 1e-9
              for o, p, _ in perfect.values()))

    biased = R.tiers(df, y + 5.0)
    m = R.change_metrics(*biased["B_local"][:2])
    check("a constant 5 K bias is not penalised (it cancels in a counterfactual)",
          abs(m["r2"] - 1) < 1e-9 and m["rmse"] < 1e-9, f"R2 {m['r2']:.6f}")

    place_only = 310 + df["place"].to_numpy()             # knows which places are hot, nothing else
    m = R.change_metrics(*R.tiers(df, place_only)["B_local"][:2])
    check("a model that only knows which places are hot scores no skill on change",
          m["r2"] <= 0.0 + 1e-9, f"R2 {m['r2']:.3f}")

    # tracks each day's weather exactly (the observed date mean) and knows nothing local
    weather_only = df.groupby("date")["lst"].transform("mean").to_numpy()
    ma = R.change_metrics(*R.tiers(df, weather_only)["A_total"][:2])
    mb = R.change_metrics(*R.tiers(df, weather_only)["B_local"][:2])
    check("weather alone looks skilful on Tier A ...", ma["r"] > 0.8, f"r {ma['r']:.3f}")
    check("... and is stripped out by Tier B", not (mb.get("r2", 0) > 0.01), f"R2 {mb.get('r2')}")

    # true response is -8*ndvi - 20*albedo; adding half of it back leaves a model that reacts half as much
    half = y + 0.5 * (8.0 * df["ndvi"].to_numpy() + 20.0 * df["albedo"].to_numpy())
    m = R.change_metrics(*R.tiers(df, half)["C_surface"][:2])
    check("a model that under-reacts by half gets slope ~2 on surface changes",
          1.6 < m["slope"] < 2.4, f"slope {m['slope']:.2f}")

    # ───────────── Tier C is a strict subset of real surface changes ─────────────
    t = R.tiers(df, y)
    nd, al = df.ndvi.values, df.albedo.values
    expect = int(np.sum((np.abs(nd[i2] - nd[i1]) >= R.NDVI_CHANGE) | (np.abs(al[i2] - al[i1]) >= R.ALBEDO_CHANGE)))
    check("Tier C keeps exactly the pairs whose surface changed", len(t["C_surface"][0]) == expect,
          f"{len(t['C_surface'][0])} of {len(i1)}")

    # ───────────── decomposition and bootstrap ─────────────
    dec = R.decompose_r2(y, weather_only, df.date.values)
    check("weather-only model: high between-date R2, no within-date R2",
          dec["between_date_r2"] > 0.9 and dec["within_date_r2"] < 0.01, str({k: round(v, 3) for k, v in dec.items()}))
    o, p, b = R.tiers(df, half)["C_surface"]
    point = R.change_metrics(o, p)["slope"]
    lo, hi = R.block_bootstrap(o, p, b, "slope", n_boot=200)
    check("block-bootstrap interval contains the point estimate", lo <= point <= hi, f"{lo:.2f} <= {point:.2f} <= {hi:.2f}")

    # ───────────── robustness splits: no leakage either way ─────────────
    from models import features as F
    rng = np.random.default_rng(5)
    n = 6000
    years = rng.choice(np.arange(2015, 2027), n)
    sp = pd.DataFrame({"lon": 76.2 + rng.uniform(0, 0.12, n), "lat": 9.9 + rng.uniform(0, 0.12, n),
                       "date": pd.to_datetime([f"{y}-01-15" for y in years]),
                       "albedo": rng.uniform(0.05, 0.3, n), "lst": rng.normal(310, 3, n)})
    tr, va, te = F.split_temporal(sp, test_from_year=2024)
    yr = lambda d: pd.to_datetime(d["date"]).dt.year
    check("temporal: train and validation contain no scene from 2024 on",
          yr(tr).max() < 2024 and yr(va).max() < 2024)
    check("temporal: test is every 2024-2026 scene and nothing else",
          yr(te).min() >= 2024 and len(te) == int((years >= 2024).sum()))
    blk = lambda d: set((np.floor(d.lon / F.BLOCK_DEG).astype(int).astype(str) + "_"
                         + np.floor(d.lat / F.BLOCK_DEG).astype(int).astype(str)))
    check("temporal: validation blocks are disjoint from training blocks", not (blk(tr) & blk(va)))

    tr, va, tx, tin, cut = F.split_feature_extrapolation(sp, "albedo", 0.90)
    check("extrapolation: train and validation never see albedo at or above the cut",
          tr.albedo.max() < cut and va.albedo.max() < cut, f"cut {cut:.3f}")
    check("extrapolation: the extrapolated test set is all at or above the cut, the in-range set below",
          tx.albedo.min() >= cut and tin.albedo.max() < cut)
    check("extrapolation: test pixels sit in blocks training never used",
          not (blk(tr) & blk(tx)) and not (blk(tr) & blk(tin)))

    print("=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
