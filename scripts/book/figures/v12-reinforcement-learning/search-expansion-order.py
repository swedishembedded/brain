# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Which cells two searches of the same map spend their expansions on.

Left: uniform-cost search; right: A* with the Manhattan distance. Every cell that was expanded carries the number of its
expansion; mud is shaded, walls are black and the cells of the path that was returned have a heavy outline. The numbers are
those of `search_planning.expansion_orders`, computed by the chapter's routine.

Swedish Embedded AB implements solutions for planners whose search effort is counted before it is spent. If your team needs
expertise in heuristic search and route planning, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
import search_planning as sp  # noqa: E402

FONT = 'font-family="Helvetica, Arial, sans-serif"'
CW = 33
OPEN, MUD, WALL = "#f7fafc", "#e9d8a6", "#2d3748"
EXPANDED, PATH, INK, MUTED = "#bee3f8", "#2b6cb0", "#1a202c", "#4a5568"


def _text(x, y, s, size=12, anchor="start", bold=False, color=INK):
    wt = ' font-weight="bold"' if bold else ""
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{color}" text-anchor="{anchor}"{wt}>{escape(s)}</text>'


def _panel(x0, y0, title, subtitle, name):
    number, path = sp._expansion_numbers(name)
    out = [_text(x0, y0 - 34, title, 13, bold=True), _text(x0, y0 - 16, subtitle, 12, color=MUTED)]
    for r in range(sp.ROWS):
        for c in range(sp.COLS):
            x, y = x0 + c * CW, y0 + r * CW
            cell, ch = (r, c), sp.GRID[r][c]
            if ch == "#":
                fill = WALL
            elif cell in number:
                fill = EXPANDED
            elif ch == "~":
                fill = MUD
            else:
                fill = OPEN
            out.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CW}" fill="{fill}" stroke="#a0aec0" stroke-width="1"/>')
            if cell in number:
                out.append(_text(x + CW / 2, y + CW / 2 + 4, str(number[cell]), 11, "middle"))
            elif cell == sp.GOAL_CELL:
                out.append(_text(x + CW / 2, y + CW / 2 + 5, "G", 14, "middle", True))
    for cell in path:
        x, y = x0 + cell[1] * CW, y0 + cell[0] * CW
        out.append(f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{CW - 3}" height="{CW - 3}" fill="none" stroke="{PATH}" stroke-width="3"/>')
    out.append(_text(x0 + sp.START[1] * CW + CW / 2, y0 + sp.START[0] * CW - 5, "S", 13, "middle", True))
    return out


def render() -> str:
    gap = 40
    width = 2 * sp.COLS * CW + gap + 40
    height = sp.ROWS * CW + 120
    pw = sp.COLS * CW
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    for k, (name, title) in enumerate((("uniform-cost", "Uniform-cost search"), ("A*, f = g + h", "A* with f = g + h"))):
        order, _, _ = sp._run_method(name)
        out += _panel(20 + k * (pw + gap), 62, title, f"{len(order)} expansions", name)
    out.append(_text(20, height - 36, "Number in a cell: the expansion that closed it. Shaded cells are mud (cost 3), black cells are walls.",
                     11, color=MUTED))
    out.append(_text(20, height - 18, "A heavy outline marks the path that was returned (cost 14 in both); S is the start, G the goal.",
                     11, color=MUTED))
    out.append("</svg>")
    return "\n".join(out) + "\n"
