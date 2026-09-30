"""Spectral indices and LST retrieval (plan §5.3).

    NDVI = (NIR - Red) / (NIR + Red)
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    albedo = broadband from Sentinel-2 bands (Liang 2001 coefficients)
    LST from Landsat C2 L2 ST_B10 (apply scale/offset -> Kelvin), or Sobrino single-channel
         retrieval from TOA if not using the L2 product.

Keep units consistent with models/constants.py (Kelvin internally).
"""


def ndvi(nir, red):
    return (nir - red) / (nir + red)


def ndbi(swir1, nir):
    return (swir1 - nir) / (swir1 + nir)


def albedo_liang(b2, b4, b8, b11, b12):
    """Broadband shortwave albedo from Sentinel-2 SR bands (Liang 2001)."""
    raise NotImplementedError


def lst_from_landsat_l2(st_b10_dn):
    """Convert Landsat C2 L2 ST_B10 digital numbers to Kelvin (scale 0.00341802, offset 149.0)."""
    raise NotImplementedError
