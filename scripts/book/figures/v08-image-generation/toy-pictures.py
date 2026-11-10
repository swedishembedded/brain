# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The 8 x 8 pictures of the toy pipeline (`latent_diffusion_systems.edit_figure_data`): the three prototypes, the textured source and the
# codec's round trip of it, the image-to-image results at three strengths, the masked run and its composite in pixel space.
#
# Swedish Embedded AB implements solutions for image editing pipelines whose preserved regions are exact. If your team needs
# expertise in masked generation and compositing, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import latent_diffusion_systems as lds  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL, GAP, LEFT, TOP = 14, 22, 20, 30
LABEL_H = 40
ROWS = (
    (("bar", "prototype: bar"), ("box", "prototype: box"), ("slash", "prototype: slash"), ("source", "source with texture"),
     ("codec", "its codec round trip")),
    (("s0.25", "strength 0.25"), ("s0.5", "strength 0.5"), ("s1.0", "strength 1"), ("masked", "masked run, decoded"),
     ("composite", "pixel composite")),
)


def _grey(v):
    g = round(127.5 * (max(-1.0, min(1.0, v)) + 1.0))
    return f"#{g:02x}{g:02x}{g:02x}"


def render() -> str:
    imgs = lds.edit_figure_data()
    side = 8 * CELL
    w = LEFT * 2 + 5 * side + 4 * GAP
    h = TOP + 2 * (side + LABEL_H)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">', f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
    for r, row in enumerate(ROWS):
        y0 = TOP + r * (side + LABEL_H) - 8
        for c, (key, label) in enumerate(row):
            x0 = LEFT + c * (side + GAP)
            img = imgs[key]
            for y in range(8):
                for x in range(8):
                    out.append(f'<rect x="{x0 + x * CELL}" y="{y0 + y * CELL}" width="{CELL}" height="{CELL}" fill="{_grey(img[y][x])}"/>')
            out.append(f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" fill="none" stroke="#4a5568" stroke-width="1"/>')
            out.append(f'<text x="{fmt(x0 + side / 2)}" y="{fmt(y0 + side + 18)}" {FONT} font-size="12" fill="#1a202c" '
                       f'text-anchor="middle">{label}</text>')
    out.append("</svg>\n")
    return "\n".join(out)
