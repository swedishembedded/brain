# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Six 6 x 6 pictures of each source (`project_image_generator.pictures_data`): two per class from the data, from the exact field, from
# the trained pixel network at guidance 1 and 4 and from the latent system at guidance 1 and 4, all from the first two starting vectors
# of each class (the pixel rows share them).
#
# Swedish Embedded AB implements solutions for image generators whose samples are judged against a known distribution. If your team needs
# expertise in evaluating generative models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_image_generator as pig  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL, GAP, GROUP, LEFT, TOP, ROW_GAP = 12, 4, 14, 200, 44, 10
ROWS = (("data", "the data"), ("exact", "exact field, 100 RK4 steps"), ("pixel1", "pixel network, g = 1"), ("pixel4", "pixel network, g = 4"),
        ("latent1", "latent system, g = 1"), ("latent4", "latent system, g = 4"))


def _grey(v):
    g = min(255, 8 * round(127.5 * (max(-1.0, min(1.0, v)) + 1.0) / 8))
    return f"#{g:02x}{g:02x}{g:02x}"


def render() -> str:
    pics = pig.pictures_data()
    side = pig.SIDE * CELL
    colx = [LEFT + k * (side + GAP) + (k // 2) * GROUP for k in range(6)]
    w = colx[-1] + side + 20
    h = TOP + len(ROWS) * (side + ROW_GAP) + 6
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">', f'<rect width="{w}" height="{h}" fill="#ffffff"/>']
    for c, name in enumerate(pig.CLASSES):
        cx = (colx[2 * c] + colx[2 * c + 1] + side) / 2
        out.append(f'<text x="{fmt(cx)}" y="{TOP - 14}" {FONT} font-size="12" font-weight="bold" fill="#1a202c" text-anchor="middle">{name}</text>')
    for r, (key, label) in enumerate(ROWS):
        y0 = TOP + r * (side + ROW_GAP)
        out.append(f'<text x="{LEFT - 12}" y="{fmt(y0 + side / 2 + 4)}" {FONT} font-size="12" fill="#1a202c" text-anchor="end">{label}</text>')
        for k, img in enumerate(pics[key]):
            x0 = colx[k]
            paths = {}                                   # one path per grey level, a small square per pixel of that level
            for y in range(pig.SIDE):
                for x in range(pig.SIDE):
                    paths.setdefault(_grey(img[y * pig.SIDE + x]), []).append(f"M{x0 + x * CELL},{y0 + y * CELL}h{CELL}v{CELL}h-{CELL}z")
            for colour, d in sorted(paths.items()):
                out.append(f'<path d="{"".join(d)}" fill="{colour}"/>')
            out.append(f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" fill="none" stroke="#4a5568" stroke-width="1"/>')
    out.append("</svg>\n")
    return "\n".join(out)
