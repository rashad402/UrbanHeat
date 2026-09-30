"""Cloud / shadow masking (plan §5.2).

Landsat C2 L2: bit-mask QA_PIXEL (cloud, cloud shadow, cirrus, dilated cloud).
Sentinel-2 SR: mask via SCL (classes 3=shadow, 8/9/10=cloud/cirrus) or QA60.
"""


def mask_landsat_qa(image):
    """Return image with cloudy/shadow pixels masked using QA_PIXEL bits."""
    raise NotImplementedError


def mask_sentinel2_scl(image):
    """Return image with cloud/shadow pixels masked using the SCL band."""
    raise NotImplementedError
