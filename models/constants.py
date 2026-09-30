"""Physical constants and unit conventions for PIML-UrbanHeat.

UNIT POLICY (enforce everywhere):
  - Temperatures are in KELVIN internally. Convert to degC only at display/IO edges.
  - Fluxes (Rn, G, H, LE) are in W m^-2.
  - Keep this module the single source of truth; import from here, don't redefine.
"""

# Stefan-Boltzmann constant [W m^-2 K^-4]
SIGMA = 5.670374419e-8

# Air density [kg m^-3] (approx; can be made T,RH-dependent later)
RHO_AIR = 1.2

# Specific heat of air at constant pressure [J kg^-1 K^-1]
CP_AIR = 1004.0

# Von Karman constant
KAPPA = 0.41

# Celsius <-> Kelvin
KELVIN_0C = 273.15


def c_to_k(t_celsius):
    return t_celsius + KELVIN_0C


def k_to_c(t_kelvin):
    return t_kelvin - KELVIN_0C
