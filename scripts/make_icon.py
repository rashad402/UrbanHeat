"""Render the UrbanHeat brand mark as a Windows .ico, for the no-console launcher.

Run:  python scripts/make_icon.py        ->  launcher/UrbanHeat.ico

The drawing is web/planner/favicon.svg redone in numpy/Pillow, because an .ico needs raster sizes
and the project has no SVG renderer. It is the same mark: a thermal-ramp tile with its hot spot at
(12, 11) of a 36-unit square, three contour rings over it, rounded corners. The gradient stops and
ring radii below are copied from the SVG, so if the mark ever changes, change them together.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "launcher", "UrbanHeat.ico")

UNITS = 36.0                       # the SVG's viewBox
HOT = (12.0, 11.0)                 # gradient centre and ring centre, in viewBox units
RADIUS = 23.0                      # gradient radius
CORNER = 11.0                      # rect rx
RINGS = (5.5, 11.0, 17.0)
RING_WIDTH, RING_ALPHA = 1.2, 0.4
STOPS = [
    (0.00, "a50026"), (0.14, "d73027"), (0.30, "f46d43"), (0.46, "fdae61"),
    (0.60, "fee090"), (0.75, "abd9e9"), (0.88, "74add1"), (1.00, "4575b4"),
]
SIZES = [16, 24, 32, 48, 64, 128, 256]
MASTER = 1024                      # drawn large, then reduced, so small sizes are smooth


def hex_rgb(h):
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def render(px=MASTER):
    scale = px / UNITS
    ys, xs = np.mgrid[0:px, 0:px]
    # pixel centres in viewBox units
    x, y = (xs + 0.5) / scale, (ys + 0.5) / scale
    d = np.clip(np.hypot(x - HOT[0], y - HOT[1]) / RADIUS, 0.0, 1.0)   # spreadMethod=pad

    offs = np.array([s[0] for s in STOPS])
    cols = np.array([hex_rgb(s[1]) for s in STOPS], dtype=float)
    rgb = np.stack([np.interp(d, offs, cols[:, c]) for c in range(3)], axis=-1)
    img = Image.fromarray(rgb.astype(np.uint8), "RGB").convert("RGBA")

    # contour rings: white at 40% opacity
    ring = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ring)
    for r in RINGS:
        c = (HOT[0] * scale, HOT[1] * scale)
        rr = r * scale
        dr.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr],
                   outline=(255, 255, 255, int(255 * RING_ALPHA)), width=max(1, round(RING_WIDTH * scale)))
    img = Image.alpha_composite(img, ring)

    # rounded-corner clip
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, px - 1, px - 1], radius=CORNER * scale, fill=255)
    img.putalpha(mask)
    return img


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    master = render()
    # Pillow writes each requested size by resampling the image it is given
    master.resize((256, 256), Image.LANCZOS).save(OUT, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"wrote {OUT}  ({os.path.getsize(OUT):,} bytes, sizes {SIZES})")


if __name__ == "__main__":
    main()
