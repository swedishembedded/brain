# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The 8 by 8 picture of the chapter cut into 2 by 2 patches, one patch flattened and embedded, and the sequence of 16 tokens it
# becomes (`vision_transformers.patch_figure_data`).
#
# Swedish Embedded AB implements solutions for turning pictures into token sequences that a transformer can read. If your team needs
# expertise in vision transformers, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vision_transformers as vt  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL = 28
W, H = 900, 470
BLUE = "#2b6cb0"
X0, Y0 = 48, 70


def render() -> str:
    data = vt.patch_figure_data()
    img, p, grid = data["image"], data["patch"], data["grid"]
    ca, cb = data["chosen"]
    d = Diagram(W, H)
    d.panel(20, 20, 310, 300, "picture 8 x 8, patches of 2 x 2")
    for r, row in enumerate(img):
        for c, v in enumerate(row):
            d.items.append(f'<rect x="{X0 + c * CELL}" y="{Y0 + r * CELL}" width="{CELL}" height="{CELL}" fill="#ffffff" '
                           f'stroke="{FRAME}" stroke-width="1"/>')
    # the chosen patch under the numbers
    d.items.append(f'<rect x="{X0 + cb * p * CELL}" y="{Y0 + ca * p * CELL}" width="{p * CELL}" height="{p * CELL}" fill="{BLUE}" '
                   f'fill-opacity="0.18"/>')
    for r, row in enumerate(img):
        for c, v in enumerate(row):
            d.items.append(f'<text x="{fmt(X0 + (c + 0.5) * CELL)}" y="{fmt(Y0 + (r + 0.5) * CELL + 4)}" {FONT} font-size="11" '
                           f'fill="{INK}" text-anchor="middle">{v}</text>')
    for a in range(grid):
        for b in range(grid):
            hot = (a, b) == (ca, cb)
            d.items.append(f'<rect x="{X0 + b * p * CELL}" y="{Y0 + a * p * CELL}" width="{p * CELL}" height="{p * CELL}" fill="none" '
                           f'stroke="{BLUE if hot else MUTED}" stroke-width="{3 if hot else 1.6}"/>')
    d.text(X0 + 4 * CELL, Y0 + 8 * CELL + 24, ["a cell shows its code 8 i + j;", f"patch ({ca}, {cb}) is highlighted"], size=11, color=MUTED)
    ymid = Y0 + (ca * p + p / 2) * CELL
    b1 = d.box(450, ymid, 140, 112, ["flatten", "row by row", str(data["vector"])])
    b2 = d.box(620, ymid, 140, 112, ["linear map", "W: D x 4", "plus bias b"])
    b3 = d.box(790, ymid, 140, 112, [f"token {ca * grid + cb}", "D numbers", f"+ position E[{ca * grid + cb}]"], "result")
    d.arrow((X0 + (cb + 1) * p * CELL, ymid), b1["l"], color=BLUE)
    d.arrow(b1["r"], b2["l"])
    d.arrow(b2["r"], b3["l"])
    d.panel(20, 340, 860, 110, "the sequence the transformer reads: 16 tokens, row-major (token 4a + b is the patch in row a, column b)")
    cw = 50
    sx = 60
    sy = 380
    for t in range(grid * grid):
        hot = t == ca * grid + cb
        d.items.append(f'<rect x="{sx + t * cw}" y="{sy}" width="{cw}" height="46" fill="{"#ebf4ff" if hot else "#ffffff"}" '
                       f'stroke="{BLUE if hot else MUTED}" stroke-width="{2.4 if hot else 1.2}"/>')
        d.text(sx + t * cw + cw / 2, sy + 23, [str(t)], size=13, weight="bold" if hot else "normal")
    tx = sx + (ca * grid + cb) * cw + cw / 2
    d.path([b3["b"], (b3["b"][0], 330), (tx, 330), (tx, sy)], color=BLUE)
    return d.svg()
