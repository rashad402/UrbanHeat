"""Spectral indices and LST retrieval (plan §5.3).

    NDVI = (NIR - Red) / (NIR + Red)
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    albedo = broadband shortwave from surface-reflectance bands
    LST from Landsat C2 L2 ST_B10 (apply scale/offset -> Kelvin)

The NDVI/NDBI/albedo helpers are backend-agnostic: they use only +,-,*,/ so they work on
numpy arrays AND ee.Image objects (Earth Engine overloads these operators). Keep units
consistent with models/constants.py (Kelvin internally; convert to degC only for display).
"""

from .constants import KELVIN_0C

# Landsat Collection 2 Level 2 surface-temperature scaling (ST_B10 -> Kelvin).
LANDSAT_ST_SCALE = 0.00341802
LANDSAT_ST_OFFSET = 149.0
# Collection 2 Level 2 surface-reflectance scaling (SR_B* -> reflectance 0..1).
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2


def ndvi(nir, red):
    """Normalized Difference Vegetation Index."""
    return (nir - red) / (nir + red)


def ndbi(swir1, nir):
    """Normalized Difference Built-up Index."""
    return (swir1 - nir) / (swir1 + nir)


def scale_landsat_sr(dn):
    """Convert Landsat C2 L2 SR digital numbers to surface reflectance (0..1)."""
    return dn * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET


def lst_from_landsat_l2(st_b10_dn, to_celsius=False):
    """Convert Landsat C2 L2 ST_B10 digital numbers to Kelvin (or degC if requested)."""
    kelvin = st_b10_dn * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET
    return kelvin - KELVIN_0C if to_celsius else kelvin


def albedo_landsat(blue, red, nir, swir1, swir2):
    """Broadband shortwave albedo from Landsat-8 surface reflectance (Silva et al. 2016).

    Inputs are surface reflectances (0..1) for SR_B2, SR_B4, SR_B5, SR_B6, SR_B7.
    NOTE: the proposal specifies Sentinel-2 albedo (Liang 2001) — implement that variant in
    build_dataset when fusing Sentinel-2; this Landsat form is for the single-sensor slice.
    """
    return (0.300 * blue + 0.277 * red + 0.233 * nir
            + 0.143 * swir1 + 0.047 * swir2)
