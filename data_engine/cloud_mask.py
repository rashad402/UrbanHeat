"""Cloud / shadow masking (plan §5.2).

Landsat C2 L2: bit-mask QA_PIXEL (dilated cloud, cirrus, cloud, cloud shadow).
Sentinel-2 SR: mask via SCL (implement when S2 fusion is added).
"""


def mask_landsat_qa(image):
    """Return image with cloudy/shadow pixels masked using QA_PIXEL bits.

    QA_PIXEL bit meanings (Landsat C2 L2): bit1 dilated cloud, bit2 cirrus,
    bit3 cloud, bit4 cloud shadow. A pixel is kept only if all four bits are 0.
    """
    qa = image.select("QA_PIXEL")
    mask = (qa.bitwiseAnd(1 << 1).eq(0)
            .And(qa.bitwiseAnd(1 << 2).eq(0))
            .And(qa.bitwiseAnd(1 << 3).eq(0))
            .And(qa.bitwiseAnd(1 << 4).eq(0)))
    return image.updateMask(mask)


def mask_sentinel2_scl(image):
    """Return image with cloud/shadow pixels masked using the SCL band (for future S2 fusion)."""
    raise NotImplementedError
