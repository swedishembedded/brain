# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""The landscape a quality-diversity search is asked to illuminate, and the archive it builds.

Left: the fitness of the behaviour space on the 10 x 10 grid, the highest value of the landscape inside each cell (found on
an 8 x 8 grid of points that includes the cell's lower edges and not its upper ones, with no effort penalty): a broad low peak in the lower left and a narrow high one in
the upper right. Right: the archive of the first MAP-Elites run of `search_planning.quality_diversity_results`, the fitness
of the elite of each cell; an empty cell is white. Both use the same colour scale.

Swedish Embedded AB implements solutions for search that keeps many different good solutions. If your team needs expertise in
quality-diversity search, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
import search_planning as sp  # noqa: E402

FONT = 'font-family="Helvetica, Arial, sans-serif"'
CW = 34
LO, HI = -0.1, 1.0
INK, MUTED = "#1a202c", "#4a5568"


def _colour(v):
    """White to dark blue; the value is clipped into [LO, HI]."""
    t = min(1.0, max(0.0, (v - LO) / (HI - LO)))
    a, b = (235, 244, 252), (23, 64, 120)
    return "#%02x%02x%02x" % tuple(round(a[i] + t * (b[i] - a[i])) for i in range(3))


def _text(x, y, s, size=12, anchor="start", bold=False, color=INK):
    wt = ' font-weight="bold"' if bold else ""
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{color}" text-anchor="{anchor}"{wt}>{escape(s)}</text>'


def _panel(x0, y0, title, subtitle, values):
    k = sp.QD_K
    out = [_text(x0, y0 - 34, title, 13, bold=True), _text(x0, y0 - 16, subtitle, 12, color=MUTED)]
    for cx in range(k):
        for cy in range(k):
            x, y = x0 + cx * CW, y0 + (k - 1 - cy) * CW
            v = values.get((cx, cy))
            fill = "#ffffff" if v is None else _colour(v)
            out.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CW}" fill="{fill}" stroke="#a0aec0" stroke-width="1"/>')
            if v is not None and v > 0.85:
                out.append(_text(x + CW / 2, y + CW / 2 + 4, f"{v:.2f}", 10, "middle", color="#ffffff"))
    for i in range(k):
        out.append(_text(x0 + i * CW + CW / 2, y0 + k * CW + 16, str(i), 11, "middle", color=MUTED))
        out.append(_text(x0 - 8, y0 + (k - 1 - i) * CW + CW / 2 + 4, str(i), 11, "end", color=MUTED))
    out.append(_text(x0 + k * CW / 2, y0 + k * CW + 34, "cell of the first descriptor coordinate", 11, "middle", color=MUTED))
    cy_mid = y0 + k * CW / 2
    out.append(f'<g transform="rotate(-90 {x0 - 32} {cy_mid})">'
               + _text(x0 - 32, cy_mid, "cell of the second descriptor coordinate", 11, "middle", color=MUTED) + "</g>")
    return out


def render() -> str:
    k = sp.QD_K
    landscape = {}
    for cx in range(k):
        for cy in range(k):
            landscape[(cx, cy)] = max(sp._landscape((2 * (cx + i / 8) / k - 1, 2 * (cy + j / 8) / k - 1))
                                      for i in range(8) for j in range(8))
    pw = k * CW
    left, gap = 44, 70
    width = left + 2 * pw + gap + 120
    height = pw + 130
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    out += _panel(left, 62, "The landscape", "highest sampled landscape value in each cell", landscape)
    arch = sp.archive_of_first_run()
    out += _panel(left + pw + gap, 62, "The MAP-Elites archive", f"{len(arch)} of {k * k} cells hold an elite", arch)
    bx = left + 2 * pw + gap + 28
    out.append(_text(bx - 4, 50, "fitness", 12, bold=True))
    steps = 12
    for i in range(steps):
        v = HI - (HI - LO) * (i + 0.5) / steps
        out.append(f'<rect x="{bx}" y="{62 + i * 20}" width="22" height="20" fill="{_colour(v)}" stroke="#a0aec0" stroke-width="0.5"/>')
    for v in (1.0, 0.5, 0.0):
        y = 62 + (HI - v) / (HI - LO) * steps * 20
        out.append(_text(bx + 28, y + 4, f"{v:g}", 11, color=MUTED))
    out.append("</svg>")
    return "\n".join(out) + "\n"
