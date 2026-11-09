# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Three stripe patterns of 24 pixels shrunk by 4 to 6 pixels, each by the naive bilinear sampler, by the mean of the covered pixels
# and by the antialiased triangle filter; each cell is shaded by its value and carries it. The values are `images.stripe_data`,
# the numbers of the chapter's table.
#
# Swedish Embedded AB implements solutions for preprocessing that does not invent patterns. If your team needs expertise in
# image downsampling for vision models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import images as im  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL, CH = 20, 22              # input cell width and row height
X0 = 170
GROUP_H = 4 * CH + 40
W = X0 + 24 * CELL + 20
H = 3 * GROUP_H - 12


def _row(d, y, values, factor, label, numbers):
    d.items.append(f'<text x="{X0 - 10}" y="{fmt(y + CH / 2 + 4)}" {FONT} font-size="12" fill="{INK}" text-anchor="end">{label}</text>')
    for i, v in enumerate(values):
        x = X0 + i * CELL * factor
        g = round(v)
        d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(CELL * factor)}" height="{CH}" fill="rgb({g},{g},{g})" '
                       f'stroke="{FRAME}" stroke-width="0.8"/>')
        if numbers:
            col = "#ffffff" if v < 128 else "#000000"
            d.items.append(f'<text x="{fmt(x + CELL * factor / 2)}" y="{fmt(y + CH / 2 + 4)}" {FONT} font-size="11" fill="{col}" '
                           f'text-anchor="middle">{round(v)}</text>')


def render() -> str:
    d = Diagram(W, H)
    for k, (name, row, naive, area, aa) in enumerate(im.stripe_data()):
        y = 14 + k * GROUP_H
        d.items.append(f'<text x="10" y="{y + 4}" {FONT} font-size="13" font-weight="bold" fill="{INK}">{name}</text>')
        _row(d, y + 12, row, 1, "input (24 pixels)", False)
        _row(d, y + 12 + CH, naive, 4, "naive bilinear", True)
        _row(d, y + 12 + 2 * CH, area, 4, "mean of 4 pixels", True)
        _row(d, y + 12 + 3 * CH, aa, 4, "antialiased triangle", True)
    return d.svg()
