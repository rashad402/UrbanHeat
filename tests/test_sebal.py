"""Unit tests for the SEB physics core (plan §8).

Run:  python tests/test_sebal.py

The decisive tests are NON-DEGENERACY (the residual actually depends on Ts and has one root)
and CAUSAL DIRECTION (raising albedo or vegetation must COOL the surface) — the latter is the
sign that the purely data-driven baselines get wrong because of land-cover confounding.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import sebal as S  # noqa: E402

# A representative Kochi dry-season built-up pixel (from the real dataset means).
BASE = dict(ndvi=0.25, albedo=0.13, s_down=642.0, t_air=301.8, rh=67.0, wind=1.2)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def main():
    print("SEB physics tests\n" + "=" * 60)

    # 1. Non-degeneracy: the residual must actually vary with Ts.
    print("\n1) Residual is a real function of Ts (not identically zero)")
    ts = np.array([295.0, 305.0, 315.0, 325.0])
    r = S.seb_residual(ts, **BASE)
    check("residual varies with Ts", np.ptp(r) > 50.0, f"range {np.ptp(r):.1f} W/m2")
    check("residual strictly decreasing", np.all(np.diff(r) < 0), f"{np.round(r,1)}")
    check("residual changes sign (a root exists)", r[0] > 0 > r[-1])

    # 2. Equilibrium temperature closes the budget.
    print("\n2) Equilibrium temperature closes the energy budget")
    teq = S.equilibrium_temperature(**BASE)
    res = S.seb_residual(teq, **BASE)
    check("residual ~ 0 at equilibrium", abs(float(res)) < 1e-3, f"{float(res):.2e} W/m2")
    check("equilibrium Ts physically plausible", 290 < float(teq) < 340,
          f"{float(teq)-273.15:.1f} degC")

    # 3. Flux magnitudes are physically sensible.
    print("\n3) Flux magnitudes")
    rn, g, h, le = S.seb_fluxes(teq, **BASE)
    rn, g, h, le = map(float, (rn, g, h, le))
    check("Rn positive at midday", 200 < rn < 900, f"Rn={rn:.0f} W/m2")
    check("G/Rn within 2-50%", 0.02 <= g / rn <= 0.5, f"G/Rn={g/rn:.2f}")
    check("H positive (surface warmer than air)", h > 0, f"H={h:.0f} W/m2")
    check("LE positive", le > 0, f"LE={le:.0f} W/m2")
    check("closure Rn = G+H+LE", abs(rn - (g + h + le)) < 1e-3,
          f"residual {rn-(g+h+le):.2e}")
    bowen = h / le
    check("Bowen ratio plausible for built-up", 0.5 < bowen < 6.0, f"H/LE={bowen:.2f}")

    # 4. CAUSAL DIRECTION — the whole point of the physics constraint.
    print("\n4) Counterfactual direction (what the data-driven models get wrong)")
    t_base = float(S.equilibrium_temperature(**BASE))

    cool_roof = dict(BASE, albedo=0.50)
    t_cool = float(S.equilibrium_temperature(**cool_roof))
    check("cool roof (albedo 0.13 -> 0.50) COOLS", t_cool < t_base,
          f"dT = {t_cool - t_base:+.2f} K")

    greening = dict(BASE, ndvi=0.25 + 0.30)
    t_green = float(S.equilibrium_temperature(**greening))
    check("greening (NDVI +0.30) COOLS", t_green < t_base,
          f"dT = {t_green - t_base:+.2f} K")

    # FULL-SURFACE response (whole pixel converted). Realistic for LST: white roofs measure
    # 10-20 K cooler than dark ones, tropical park cool islands reach 5-10 K.
    check("cool-roof full-surface response physical (2-10 K)",
          2.0 <= (t_base - t_cool) <= 10.0, f"{t_base - t_cool:.2f} K")
    check("greening full-surface response physical (2-12 K)",
          2.0 <= (t_base - t_green) <= 12.0, f"{t_base - t_green:.2f} K")

    # APPLIED response, after realistic partial coverage (built fraction x roof share) — this is
    # what the proposal's published field ranges describe.
    pv = float(S.fractional_vegetation(BASE["ndvi"]))
    built = 1.0 - pv
    applied_cool = (t_base - t_cool) * built * 0.5          # roofs are ~half of built area
    applied_green = (t_base - t_green) * built * 0.5        # greening a comparable share
    # The 0.5 roof/greening share here is ILLUSTRATIVE — exact coverage scaling is the
    # counterfactual layer's job, so this asserts the right order of magnitude (0.5-4 K),
    # overlapping both published ranges, rather than a precise value.
    check("applied cool-roof is order of published range (0.5-4.0 K)",
          0.5 <= applied_cool <= 4.0, f"{applied_cool:.2f} K (full-surface {t_base-t_cool:.2f})")
    check("applied greening is order of published range (0.5-4.0 K)",
          0.5 <= applied_green <= 4.0, f"{applied_green:.2f} K (full-surface {t_base-t_green:.2f})")

    # more sun -> hotter; more wind -> closer to air temperature
    t_sun = float(S.equilibrium_temperature(**dict(BASE, s_down=800.0)))
    check("more solar radiation WARMS", t_sun > t_base, f"dT = {t_sun - t_base:+.2f} K")
    t_windy = float(S.equilibrium_temperature(**dict(BASE, wind=6.0)))
    check("more wind pulls Ts toward air temp", abs(t_windy - BASE["t_air"]) < abs(t_base - BASE["t_air"]),
          f"{t_windy-273.15:.1f} vs {t_base-273.15:.1f} degC, Ta={BASE['t_air']-273.15:.1f}")

    # 5. Torch compatibility + differentiability (needed for the PINN loss).
    print("\n5) Torch tensors and gradients")
    try:
        import torch
        tts = torch.tensor([310.0], requires_grad=True, dtype=torch.float64)
        kw = {k: torch.tensor([v], dtype=torch.float64) for k, v in BASE.items()}
        rt = S.seb_residual(tts, **kw)
        rt.sum().backward()
        gr = float(tts.grad[0])
        rt_v = float(rt.detach()[0])
        check("residual computes on torch tensors", bool(torch.isfinite(rt).all()),
              f"{rt_v:.1f} W/m2")
        check("gradient flows and is negative", np.isfinite(gr) and gr < 0, f"d(res)/dTs = {gr:.2f}")
        rn_np = float(S.seb_residual(np.array([310.0]), **BASE)[0])
        check("torch and numpy agree", abs(rt_v - rn_np) < 1e-6, f"{rt_v:.4f} vs {rn_np:.4f}")
    except ImportError:
        print("  SKIP  torch not installed")

    # 6. Vectorised over the real dataset (if present).
    print("\n6) Behaviour on real data")
    pq = "data/processed/kochi_samples.parquet"
    if os.path.exists(pq):
        import pandas as pd
        df = pd.read_parquet(pq).sample(4000, random_state=0)
        teq = S.equilibrium_temperature(df.ndvi.values, df.albedo.values, df.s_down.values,
                                        df.t_air.values, df.rh.values, df.wind.values)
        obs = df.lst.values
        bias = float(np.mean(teq - obs))
        corr = float(np.corrcoef(teq, obs)[0, 1])
        check("equilibrium Ts is finite everywhere", np.all(np.isfinite(teq)))
        # This is a PURE first-principles prediction (no fitting to LST) scored against noisy
        # 30 m satellite retrievals, so a modest correlation is expected and acceptable; closing
        # the remaining gap is exactly what the PINN's data term is for. Absolute bias matters
        # more here, because a biased physics term would drag the PINN's predictions off.
        check("equilibrium Ts correlates with observed LST", corr > 0.2,
              f"r={corr:.3f}")
        check("equilibrium Ts is unbiased (|bias| < 1.5 K)", abs(bias) < 1.5,
              f"mean bias {bias:+.2f} K")
        print(f"        observed  LST mean {obs.mean()-273.15:.1f} degC")
        print(f"        SEB-closure Ts mean {teq.mean()-273.15:.1f} degC")
    else:
        print("  SKIP  dataset not built")

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
