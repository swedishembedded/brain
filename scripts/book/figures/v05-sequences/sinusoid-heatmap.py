# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The sinusoidal encoding of 32 positions at width 32 (base 10000) as a grid: one row for each position, one column for each
# channel, darker blue for values near 1 and darker orange for values near -1. The left columns oscillate quickly with the position,
# the right columns hardly move. Values from `positions.sinusoid_grid_data`.
#
# Swedish Embedded AB implements solutions for long-context transformer models, from the position encoding to the kernels. If your
# team needs expertise in positional encodings, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import positions as ps  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402
from svgplot import fmt  # noqa: E402

POS, DIM = 32, 32
CELL = 13
GX, GY = 74, 62


def _shade(v):
    """Orange at -1, white at 0, blue at +1."""
    pos, neg, mid = (43, 108, 176), (192, 86, 33), (255, 255, 255)
    end = pos if v >= 0 else neg
    t = abs(v)
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(mid, end))


def render() -> str:
    grid = ps.sinusoid_grid_data(POS, DIM)
    w, h = GX + DIM * CELL + 150, GY + POS * CELL + 14
    d = Diagram(w, h)
    d.text(GX + DIM * CELL / 2, 18, ["Sinusoidal encoding, width 32, base 10000"], size=14, weight="bold")
    d.text(GX + DIM * CELL / 2, 36, ["channel (pairs: sin, cos at the rate w_m)"], size=12, color=MUTED)
    for c in range(0, DIM, 4):
        d.text(GX + (c + 0.5) * CELL, GY - 5, [str(c)], size=11)
    for r in range(0, POS, 4):
        d.text(GX - 8, GY + (r + 0.5) * CELL + 4, [str(r)], size=11, anchor="end")
    d.text(GX - 52, GY + POS * CELL / 2, ["position"], size=12, color=MUTED, anchor="middle")
    for r in range(POS):
        for c in range(DIM):
            d.items.append(f'<rect x="{fmt(GX + c * CELL)}" y="{fmt(GY + r * CELL)}" width="{CELL}" height="{CELL}" '
                           f'fill="{_shade(grid[r][c])}"/>')
    lx = GX + DIM * CELL + 24
    for k, v in enumerate((1.0, 0.5, 0.0, -0.5, -1.0)):
        y = GY + 20 + k * 26
        d.items.append(f'<rect x="{lx}" y="{y}" width="22" height="22" fill="{_shade(v)}" stroke="#a0aec0" stroke-width="0.5"/>')
        d.text(lx + 30, y + 16, [f"{v:+.1f}"], size=12, color=INK, anchor="start")
    return d.svg()
