"""Calibrate the SEB parameterization to Kochi (plan §8 — the humid-tropical coastal novelty).

Run:  python scripts/calibrate_sebal.py

Three knobs are swept against the real dataset:
  R_FREE       free-convection resistance [s/m]. Neutral-stability r_ah diverges at low wind,
               which made the SEB temperature track ERA5 wind (a date-level 11 km field) rather
               than land cover. Free convection acts in parallel and bounds it.
  REX_URBAN    excess resistance over built-up [s/m]. Landsat gives RADIOMETRIC surface
               temperature while the flux equations assume AERODYNAMIC temperature; without this
               series term the closure runs several K cold over cities.
  F_MOIST_MAX  moisture availability of dense vegetation — sets how strongly greening cools and
               the Bowen ratio.

SCORING — in priority order:
  1. |bias| small.  A biased physics term would drag the PINN's predictions off, so absolute
     agreement matters more than correlation here.
  2. correlation with observed LST (overall and within-date) and a weak wind artefact.
  3. counterfactual DIRECTION correct (cooling). Note the full-surface magnitudes are large by
     design (see the note in models/sebal.py) — do NOT tune them down to the published
     partial-coverage ranges; that scaling belongs in the counterfactual layer.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import sebal as S  # noqa: E402

PARQUET = "data/processed/kochi_samples.parquet"
BASE = dict(ndvi=0.25, albedo=0.13, s_down=642.0, t_air=301.8, rh=67.0, wind=1.2)


def score(df):
    teq = S.equilibrium_temperature(df.ndvi.values, df.albedo.values, df.s_down.values,
                                    df.t_air.values, df.rh.values, df.wind.values)
    obs = df.lst.values
    d = df.assign(teq=teq)
    within = d.groupby("date").apply(
        lambda g: np.corrcoef(g.lst, g.teq)[0, 1] if len(g) > 50 else np.nan,
        include_groups=False)
    t0 = float(S.equilibrium_temperature(**BASE))
    return {
        "bias": float(teq.mean() - obs.mean()),
        "corr": float(np.corrcoef(obs, teq)[0, 1]),
        "within": float(np.nanmedian(within)),
        "std_ratio": float(teq.std() / obs.std()),
        "c_wind": float(np.corrcoef(teq, df.wind)[0, 1]),
        "dT_cool": t0 - float(S.equilibrium_temperature(**dict(BASE, albedo=0.50))),
        "dT_green": t0 - float(S.equilibrium_temperature(**dict(BASE, ndvi=0.55))),
    }


def main():
    df = pd.read_parquet(PARQUET).sample(20000, random_state=0)
    print(f"Observed: mean {df.lst.mean()-273.15:.1f} C, std {df.lst.std():.2f} K, "
          f"corr(LST,wind) {np.corrcoef(df.lst, df.wind)[0,1]:+.3f}\n")
    print(f"{'R_FREE':>7}{'REXurb':>7}{'Fmoist':>7}{'bias':>7}{'corr':>7}{'within':>8}"
          f"{'std_r':>7}{'c_wind':>8}{'dTcool':>8}{'dTgrn':>7}  ok")

    keep = []
    for r_free in [40, 60, 80, 120]:
        for rex in [20, 30, 40, 60]:
            for f_max in [0.5, 0.6, 0.7]:
                S.R_FREE, S.REX_URBAN, S.F_MOIST_MAX = float(r_free), float(rex), float(f_max)
                r = score(df)
                ok = abs(r["bias"]) < 1.0 and r["dT_cool"] > 0 and r["dT_green"] > 0
                print(f"{r_free:>7}{rex:>7}{f_max:>7.1f}{r['bias']:>7.2f}{r['corr']:>7.3f}"
                      f"{r['within']:>8.3f}{r['std_ratio']:>7.2f}{r['c_wind']:>8.3f}"
                      f"{r['dT_cool']:>8.2f}{r['dT_green']:>7.2f}  {'Y' if ok else '.'}")
                if ok:
                    # among unbiased options prefer correlation and a weak wind artefact
                    keep.append((r["corr"] + r["within"] - abs(r["c_wind"]),
                                 r_free, rex, f_max, r))

    if keep:
        _, r_free, rex, f_max, r = max(keep)
        print(f"\nBEST: R_FREE={r_free}, REX_URBAN={rex}, F_MOIST_MAX={f_max}")
        print(f"  bias {r['bias']:+.2f} K | corr {r['corr']:.3f} (within-date {r['within']:.3f})"
              f" | wind artefact {r['c_wind']:+.3f}")
        print(f"  full-surface cooling: cool roof {r['dT_cool']:.2f} K, greening {r['dT_green']:.2f} K")
        print("\nSet these in models/sebal.py.")
    else:
        print("\nNo combination met the bias constraint.")


if __name__ == "__main__":
    main()
