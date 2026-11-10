# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A 3 by 3 kernel over the 5 by 5 image of the chapter padded with a border of one zero (stride 1): the window that produces output
# (1, 2) and the one that produced (0, 0), and the 5 by 5 output (`convolution.sliding_data`, `convolution.SOBEL`).
#
# Swedish Embedded AB implements solutions for convolution kernels that match a reference. If your team needs expertise in
# convolutional networks, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import convolution as cv  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL = 38
BLUE, GREEN = "#2b6cb0", "#2f855a"
W, H = 800, 420


def _cells(d, x0, y0, values, muted_border=False, fill="#ffffff"):
    for r, row in enumerate(values):
        for c, v in enumerate(row):
            border = muted_border and (r in (0, len(values) - 1) or c in (0, len(row) - 1))
            d.items.append(f'<rect x="{fmt(x0 + c * CELL)}" y="{fmt(y0 + r * CELL)}" width="{CELL}" height="{CELL}" '
                           f'fill="{"#edf2f7" if border else fill}" stroke="{FRAME}" stroke-width="1"/>')
            d.items.append(f'<text x="{fmt(x0 + (c + 0.5) * CELL)}" y="{fmt(y0 + (r + 0.5) * CELL + 4)}" {FONT} font-size="13" '
                           f'fill="{MUTED if border else INK}" text-anchor="middle">{v}</text>')


def _window(d, x0, y0, r, c, color, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    d.items.append(f'<rect x="{fmt(x0 + c * CELL)}" y="{fmt(y0 + r * CELL)}" width="{3 * CELL}" height="{3 * CELL}" fill="{color}" '
                   f'fill-opacity="0.10" stroke="{color}" stroke-width="3"{extra}/>')


def _mark(d, x0, y0, r, c, color, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    d.items.append(f'<rect x="{fmt(x0 + c * CELL + 1.5)}" y="{fmt(y0 + r * CELL + 1.5)}" width="{CELL - 3}" height="{CELL - 3}" fill="{color}" '
                   f'fill-opacity="0.15" stroke="{color}" stroke-width="3"{extra}/>')


def render() -> str:
    padded, out = cv.sliding_data()
    d = Diagram(W, H)
    px, py = 40, 80
    d.panel(20, 20, 7 * CELL + 40, 7 * CELL + 105, "image X with a border of zeros (padding 1)")
    _cells(d, px, py, padded, muted_border=True)
    _window(d, px, py, 0, 0, GREEN, "6 3")
    _window(d, px, py, 1, 2, BLUE)
    ox, oy = 440, 80
    d.panel(ox - 20, 20, 5 * CELL + 40 + 140, 5 * CELL + 105, "output Y (5 by 5)")
    _cells(d, ox, oy, out)
    _mark(d, ox, oy, 0, 0, GREEN, "6 3")
    _mark(d, ox, oy, 1, 2, BLUE)
    kx, ky = 440 + 5 * CELL + 30, 120
    d.text(kx + 1.5 * CELL, ky - 20, ["kernel K"], size=13, weight="bold")
    _cells(d, kx, ky, cv.SOBEL)
    d.text(kx + 1.5 * CELL, ky + 3 * CELL + 22, ["taps are multiplied by", "the pixels under them", "and summed"], size=12, color=MUTED)
    d.text(px + 3.5 * CELL, py + 7 * CELL + 28, ["solid blue window gives the blue output cell (1, 2);", "dashed green window gives the green output cell (0, 0)"], size=12, color=MUTED)
    return d.svg()
