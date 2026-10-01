"""Surface Energy Balance (SEBAL-style) flux parameterizations — the physics core (plan §8).

Energy balance, all fluxes in W m^-2, temperatures in KELVIN:

    Rn = (1 - alpha)*S_down + eps*L_down - eps*sigma*Ts^4
    G  = Rn * Gamma(NDVI, alpha, Ts)                      (Bastiaanssen 1998)
    H  = rho*cp*(Ts - Ta) / r_ah(u, NDVI)
    LE = alpha_PT * f_moist(NDVI) * (D/(D+gamma)) * (Rn - G)     <-- INDEPENDENT (Priestley-Taylor)

    seb_residual(Ts) = Rn(Ts) - G(Ts) - H(Ts) - LE(Ts)

WHY LE IS NOT THE RESIDUAL
    The proposal defines LE = Rn - G - H, which makes |Rn - G - H - LE| identically zero: the
    physics loss would constrain nothing. Here LE is estimated INDEPENDENTLY via Priestley-Taylor
    from air temperature, humidity and available energy, so the residual is a real function of Ts.
    Because Rn falls with Ts (-eps*sigma*Ts^4) while H rises with Ts, the residual is strictly
    decreasing in Ts and has exactly ONE root - the temperature that closes the budget. That root
    is what the PINN's physics term pulls its prediction towards.

CORRECTION TO THE PROPOSAL'S Rn
    The proposal writes Rn = (1-a)S + eps*L_down - (1-eps)L_down - eps*sigma*Ts^4, which
    double-counts longwave. Starting from Rn = (1-a)S + L_down - L_up with
    L_up = eps*sigma*Ts^4 + (1-eps)*L_down gives Rn = (1-a)S + eps*L_down - eps*sigma*Ts^4,
    which is what is implemented. Worth fixing in the report text.

CALIBRATION KNOBS (the project's "humid-tropical coastal" novelty lives here)
    F_MOIST_MIN/MAX  - how much of Priestley-Taylor potential evaporation a built vs vegetated
                       surface actually realises. These set the Bowen ratio (H/LE); Kerala's
                       humid coast should land below ~0.5 for vegetated surfaces.
    Z0M_URBAN        - roughness length of built-up surface, which sets r_ah and hence H.
    Calibrate both against ERA5-Land latent-heat flux for the AOI (a concrete follow-up task).

All functions accept numpy arrays, python floats, or torch tensors (so the same code serves
dataset diagnostics and the differentiable training loss).
"""

import numpy as np

try:
    import torch
except ImportError:                                   # torch optional for numpy-only use
    torch = None

# ---- constants -------------------------------------------------------------
SIGMA = 5.670374419e-8      # Stefan-Boltzmann [W m^-2 K^-4]
RHO_AIR = 1.2               # air density [kg m^-3]
CP_AIR = 1004.0             # specific heat of air [J kg^-1 K^-1]
KAPPA = 0.41                # von Karman
KELVIN_0C = 273.15

GAMMA_HPA = 0.67            # psychrometric constant [hPa K^-1] at sea level
ALPHA_PT = 1.26             # Priestley-Taylor coefficient (saturated surface)
# Moisture availability — calibrated for Kochi (scripts/calibrate_sebal.py).
F_MOIST_MIN = 0.25          # built/bare surface
F_MOIST_MAX = 0.60          # dense vegetation

Z0M_URBAN = 0.50            # roughness length of built-up surface [m]
Z_WIND = 10.0               # ERA5 wind reference height [m]
Z1, Z2 = 0.1, 2.0           # SEBAL near-surface heights for r_ah [m]
# Neutral-stability r_ah diverges at low wind, which made the SEB temperature track ERA5 wind
# (a date-level, 11 km field) instead of land cover. Physically, strong daytime heating drives
# FREE convection in parallel with forced convection, bounding the resistance:
#     1/r_ah = 1/r_neutral + 1/R_FREE
# R_FREE is the dominant calibration knob for Kochi — see scripts/calibrate_sebal.py.
R_FREE = 60.0               # free-convection resistance [s m^-1]
RAH_MIN, RAH_MAX = 5.0, 300.0
WIND_MIN, WIND_MAX = 0.5, 20.0

# EXCESS RESISTANCE (kB^-1 effect). Landsat measures RADIOMETRIC surface temperature, but the
# turbulent-flux equations assume AERODYNAMIC temperature; radiometric LST of sunlit surfaces
# runs hotter. Without this term the closure cannot reach observed urban LST (it ran ~5 K cold)
# without making r_ah so large that ERA5 wind dominates the pattern. The excess resistance is
# added in SERIES, is land-cover dependent (large over built-up, small over vegetation) and is
# wind-independent, so it raises urban Ts without reintroducing the wind artefact.
REX_URBAN = 30.0            # excess resistance over built-up/bare [s m^-1]
REX_VEG = 5.0               # excess resistance over dense vegetation [s m^-1]

# NOTE ON COUNTERFACTUAL MAGNITUDES
#   These fluxes give a FULL-SURFACE response: converting an entire pixel's albedo 0.13 -> 0.50
#   cools LST by ~6 K, and full conversion to vegetation by ~7 K. Those are realistic for
#   *surface* temperature (white roofs measure 10-20 K cooler than dark ones; tropical park cool
#   islands reach 5-10 K in LST). The proposal's published 1.5-4.0 K / 0.5-3.0 K ranges describe
#   PARTIAL-COVERAGE, area-averaged (often air) temperature effects. The coverage scaling lives
#   in the counterfactual layer (built fraction x roof share), not here — do not "calibrate" these
#   full-surface sensitivities down to the published ranges.

NDVI_SOIL, NDVI_VEG = 0.2, 0.5


# ---- backend-agnostic helpers ---------------------------------------------
def _is_t(x):
    return torch is not None and isinstance(x, torch.Tensor)


def _exp(x):
    return torch.exp(x) if _is_t(x) else np.exp(x)


def _clip(x, lo, hi):
    return torch.clamp(x, lo, hi) if _is_t(x) else np.clip(x, lo, hi)


def _log(x):
    return torch.log(x) if _is_t(x) else np.log(x)


# ---- humidity / radiation --------------------------------------------------
def saturation_vapour_pressure(t_kelvin):
    """Saturation vapour pressure e_s [hPa] (Magnus/Tetens)."""
    tc = t_kelvin - KELVIN_0C
    return 6.112 * _exp(17.67 * tc / (tc + 243.5))


def svp_slope(t_kelvin):
    """Slope of the saturation vapour pressure curve, Delta [hPa K^-1]."""
    tc = t_kelvin - KELVIN_0C
    return 4098.0 * saturation_vapour_pressure(t_kelvin) / (tc + 237.3) ** 2


def actual_vapour_pressure(t_air, rh):
    """Actual vapour pressure e_a [hPa] from air temperature [K] and RH [%]."""
    return (rh / 100.0) * saturation_vapour_pressure(t_air)


def fractional_vegetation(ndvi):
    """Fractional vegetation cover Pv in [0,1] (NDVI threshold method)."""
    pv = (ndvi - NDVI_SOIL) / (NDVI_VEG - NDVI_SOIL)
    return _clip(pv, 0.0, 1.0) ** 2


def emissivity_from_ndvi(ndvi):
    """Broadband surface emissivity, ~0.986 (bare) to ~0.990 (vegetated)."""
    return 0.986 + 0.004 * fractional_vegetation(ndvi)


def incoming_longwave(t_air, rh):
    """Downwelling longwave L_down [W m^-2] (Brutsaert 1975 clear-sky emissivity)."""
    e_a = actual_vapour_pressure(t_air, rh)
    eps_air = 1.24 * (e_a / t_air) ** (1.0 / 7.0)
    return eps_air * SIGMA * t_air ** 4


def net_radiation(albedo, s_down, l_down, eps, ts):
    """Rn [W m^-2]. See the correction note in the module docstring."""
    return (1.0 - albedo) * s_down + eps * l_down - eps * SIGMA * ts ** 4


# ---- turbulent / ground fluxes --------------------------------------------
def roughness_length(ndvi):
    """Momentum roughness length z0m [m]: blend of vegetation (SEBAL) and urban roughness."""
    pv = fractional_vegetation(ndvi)
    z0_veg = 0.005 + 0.5 * _clip(ndvi / 0.8, 0.0, 1.0) ** 2.5
    return (1.0 - pv) * Z0M_URBAN + pv * z0_veg


def excess_resistance(ndvi):
    """Land-cover dependent excess resistance for heat (kB^-1 effect) [s m^-1]."""
    pv = fractional_vegetation(ndvi)
    return REX_URBAN * (1.0 - pv) + REX_VEG * pv


def aerodynamic_resistance(wind, ndvi):
    """r_ah [s m^-1]: forced and free convection in parallel, plus excess resistance in series."""
    z0m = roughness_length(ndvi)
    u = _clip(wind, WIND_MIN, WIND_MAX)
    u_star = KAPPA * u / _log(Z_WIND / z0m)
    r_neutral = _log(Z2 / Z1) / (KAPPA * u_star)
    r_transport = 1.0 / (1.0 / r_neutral + 1.0 / R_FREE)   # parallel conductances
    return _clip(r_transport + excess_resistance(ndvi), RAH_MIN, RAH_MAX)


def sensible_heat_flux(ts, t_air, wind, ndvi):
    """H [W m^-2] = rho*cp*(Ts - Ta)/r_ah."""
    return RHO_AIR * CP_AIR * (ts - t_air) / aerodynamic_resistance(wind, ndvi)


def ground_heat_flux(rn, ndvi, albedo, ts):
    """G [W m^-2] as a fraction of Rn (Bastiaanssen 1998 SEBAL), clamped to a sane range."""
    tc = ts - KELVIN_0C
    a = _clip(albedo, 0.05, 0.6)
    gamma = (tc / a) * (0.0038 * a + 0.0074 * a ** 2) * (1.0 - 0.98 * _clip(ndvi, 0.0, 1.0) ** 4)
    return rn * _clip(gamma, 0.02, 0.5)


def moisture_availability(ndvi):
    """f_moist in [F_MOIST_MIN, F_MOIST_MAX] — the fraction of PT potential actually realised."""
    return F_MOIST_MIN + (F_MOIST_MAX - F_MOIST_MIN) * fractional_vegetation(ndvi)


def latent_heat_priestley_taylor(rn, g, t_air, ndvi):
    """INDEPENDENT LE estimate [W m^-2] — this is what makes the physics residual non-degenerate.

        LE = alpha_PT * f_moist(NDVI) * Delta/(Delta + gamma) * (Rn - G)
    """
    delta = svp_slope(t_air)
    return ALPHA_PT * moisture_availability(ndvi) * (delta / (delta + GAMMA_HPA)) * (rn - g)


# ---- the residual used by the PINN physics loss ----------------------------
def seb_fluxes(ts, ndvi, albedo, s_down, t_air, rh, wind):
    """Return (Rn, G, H, LE) at the given surface temperature."""
    eps = emissivity_from_ndvi(ndvi)
    l_down = incoming_longwave(t_air, rh)
    rn = net_radiation(albedo, s_down, l_down, eps, ts)
    g = ground_heat_flux(rn, ndvi, albedo, ts)
    h = sensible_heat_flux(ts, t_air, wind, ndvi)
    le = latent_heat_priestley_taylor(rn, g, t_air, ndvi)
    return rn, g, h, le


def seb_residual(ts, ndvi, albedo, s_down, t_air, rh, wind):
    """Surface energy balance residual [W m^-2]: Rn - G - H - LE.

    Strictly decreasing in ts, so it has a single root (the energy-closing temperature).
    This is the quantity squared and averaged in the PINN's physics loss.
    """
    rn, g, h, le = seb_fluxes(ts, ndvi, albedo, s_down, t_air, rh, wind)
    return rn - g - h - le


def equilibrium_temperature(ndvi, albedo, s_down, t_air, rh, wind,
                            lo=250.0, hi=360.0, iters=60):
    """Solve seb_residual(Ts) = 0 by bisection (numpy only) — diagnostics and tests.

    Returns the surface temperature [K] that closes the energy budget.
    """
    lo = np.full_like(np.asarray(ndvi, dtype="float64"), lo)
    hi = np.full_like(np.asarray(ndvi, dtype="float64"), hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = seb_residual(mid, ndvi, albedo, s_down, t_air, rh, wind)
        # residual decreasing in Ts: r > 0 means mid is too cold
        lo = np.where(r > 0, mid, lo)
        hi = np.where(r > 0, hi, mid)
    return 0.5 * (lo + hi)
