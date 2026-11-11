# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The first test picture of the texture world (`restoration.sr_pictures_data`): the picture, its 2 x 2 box mean, the output of the
# trained network (the best squared-error restorer: no texture) and the same output with a texture drawn like the real one added.
#
# Swedish Embedded AB implements solutions for super-resolution whose loss of detail is measured rather than assumed. If your team
# needs expertise in image restoration, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import restoration  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL, GAP, LEFT, TOP, LABEL_H = 20, 30, 20, 20, 44


def _grey(v, lo, hi):
    g = round(255 * (max(lo, min(hi, v)) - lo) / (hi - lo))
    return f"#{g:02x}{g:02x}{g:02x}"


def render() -> str:
    x, low, out, smp = restoration.sr_pictures_data()
    lo = min(min(r) for r in x)
    hi = max(max(r) for r in x)
    panels = ((x, 8, "the picture"), (low, 4, "its 2 x 2 means"), (out, 8, "squared-error network"), (smp, 8, "network plus a sampled texture"))
    side = 8 * CELL
    w = LEFT * 2 + 4 * side + 3 * GAP
    h = TOP + side + LABEL_H
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">', f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
    for k, (img, n, label) in enumerate(panels):
        x0 = LEFT + k * (side + GAP)
        cell = side // n
        for yy in range(n):
            for xx in range(n):
                svg.append(f'<rect x="{x0 + xx * cell}" y="{TOP + yy * cell}" width="{cell}" height="{cell}" fill="{_grey(img[yy][xx], lo, hi)}"/>')
        svg.append(f'<rect x="{x0}" y="{TOP}" width="{side}" height="{side}" fill="none" stroke="#4a5568" stroke-width="1"/>')
        svg.append(f'<text x="{fmt(x0 + side / 2)}" y="{fmt(TOP + side + 20)}" {FONT} font-size="12" fill="#1a202c" text-anchor="middle">{label}</text>')
    svg.append("</svg>\n")
    return "\n".join(svg)
