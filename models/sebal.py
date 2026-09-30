"""Surface Energy Balance (SEBAL-style) flux parameterizations.

This is the PHYSICS CORE of the project (implementation plan §8). Build and unit-test
these pure functions BEFORE wiring them into the PINN loss.

Energy balance (all fluxes in W m^-2, temperatures in Kelvin):

    Rn = (1 - alpha) * S_down + eps * L_down - (1 - eps) * L_down - eps * SIGMA * Ts^4
    G  = Rn * Gamma(NDVI, alpha, Ts)
    H  = RHO_AIR * CP_AIR * (Ts - Ta) / r_ah(u)
    LE = <independent estimate>            # NOT simply Rn - G - H (see design note below)

DESIGN NOTE (critical — plan §8):
    The proposal writes LE = Rn - G - H (latent heat as the SEB residual). If the physics
    loss then penalizes |Rn - G - H - LE|^2, it is IDENTICALLY ZERO for any Ts and constrains
    nothing. To make the physics term meaningful, estimate LE INDEPENDENTLY (e.g.
    Priestley-Taylor / Penman-Monteith using RH and available energy) and penalize the
    network's Ts against the temperature that closes  Rn(Ts) = G + H(Ts) + LE_independent.
    Resolve the exact formulation with the supervisor before training (plan §8, action item).

All functions accept numpy arrays or torch tensors (keep them backend-agnostic where possible
so the same code serves both dataset diagnostics and the differentiable training loss).
"""

from .constants import SIGMA, RHO_AIR, CP_AIR


def emissivity_from_ndvi(ndvi):
    """Surface emissivity via the NDVI-threshold method. Returns eps in ~[0.95, 0.99]."""
    raise NotImplementedError


def incoming_longwave(t_air, rh):
    """Downwelling longwave L_down [W m^-2] (e.g. Brutsaert clear-sky emissivity)."""
    raise NotImplementedError


def net_radiation(albedo, s_down, l_down, eps, ts):
    """Rn [W m^-2]. ts in Kelvin."""
    raise NotImplementedError


def ground_heat_flux(rn, ndvi, albedo, ts):
    """G [W m^-2] as a fraction Gamma(NDVI, alpha, Ts) of Rn (Bastiaanssen 1998)."""
    raise NotImplementedError


def aerodynamic_resistance(wind):
    """r_ah [s m^-1] as a function of wind speed (humid-coastal calibration lives here)."""
    raise NotImplementedError


def sensible_heat_flux(ts, t_air, wind):
    """H [W m^-2] = rho*cp*(Ts - Ta) / r_ah(u)."""
    raise NotImplementedError


def latent_heat_independent(rn, g, t_air, rh):
    """INDEPENDENT LE estimate [W m^-2] (Priestley-Taylor / Penman-Monteith).

    This is what makes the SEB residual non-degenerate. See DESIGN NOTE above.
    """
    raise NotImplementedError


def seb_residual(ts, features):
    """Surface energy balance residual [W m^-2] used by the PINN physics loss.

    residual = Rn(ts) - G - H(ts) - LE_independent

    Args:
        ts: predicted land surface temperature [K] (torch tensor during training).
        features: dict/array of {ndvi, ndbi, albedo, s_down, t_air, rh, wind}.

    Returns:
        residual with the same shape as ts.
    """
    raise NotImplementedError
