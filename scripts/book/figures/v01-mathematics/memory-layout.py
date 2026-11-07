# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A 3x4 matrix and the two ways to lay it out as one line of numbers.

Cell (i, j) holds the number 10 i + j and is coloured by its row. Row-major
memory keeps each row together; column-major memory interleaves the rows. The
memory order is computed with the chapter's worked module, so the picture and
the tables cannot disagree.

Swedish Embedded AB implements solutions for memory layout and data movement in
numerical software. If your team needs expertise in how arrays map to memory,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
import arrays as ar  # noqa: E402

FONT = 'font-family="Helvetica, Arial, sans-serif"'
FILLS = ["#bee3f8", "#fed7aa", "#c6f6d5"]
STROKES = ["#2b6cb0", "#c05621", "#2f855a"]


def cell(x, y, w, h, label, row, size=13):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{FILLS[row]}" stroke="{STROKES[row]}"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + 4.5}" {FONT} font-size="{size}" fill="#1a202c" '
            f'text-anchor="middle">{escape(str(label))}</text>')


def text(x, y, s, size=12, anchor="start", bold=False, color="#1a202c"):
    wt = ' font-weight="bold"' if bold else ""
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{color}" text-anchor="{anchor}"{wt}>{escape(s)}</text>'


def render() -> str:
    rows, cols = 3, 4
    A = ar.Arr([10 * i + j for i in range(rows) for j in range(cols)], (rows, cols))
    n = rows * cols
    cw = 46
    left = 70
    width = left + n * cw + 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 400" width="{width}" height="400">',
           f'<rect width="{width}" height="400" fill="#ffffff"/>']
    out.append(text(left, 22, "A, shape [3, 4]: entry (i, j) holds 10 i + j; colour = row i", 13, bold=True))
    mx, my = left, 36
    for i in range(rows):
        out.append(text(mx - 8, my + i * cw + cw / 2 + 4, f"i={i}", 11, "end", color="#4a5568"))
        for j in range(cols):
            out.append(cell(mx + j * cw, my + i * cw, cw, cw, A.at((i, j)), i))
    for j in range(cols):
        out.append(text(mx + j * cw + cw / 2, my + rows * cw + 14, f"j={j}", 11, "middle", color="#4a5568"))

    def strip(y, title, order, formula):
        out.append(text(left, y - 8, title, 13, bold=True))
        out.append(text(left + n * cw, y - 8, formula, 12, "end", color="#4a5568"))
        for k, (i, j) in enumerate(order):
            out.append(cell(left + k * cw, y, cw, cw, A.at((i, j)), i))
            out.append(text(left + k * cw + cw / 2, y + cw + 15, str(k), 11, "middle", color="#4a5568"))
        out.append(text(left - 8, y + cw / 2 + 4, "memory", 11, "end", color="#4a5568"))
        out.append(text(left - 8, y + cw + 15, "offset", 11, "end", color="#4a5568"))

    rs, cs = ar.row_major_strides(A.shape), ar.col_major_strides(A.shape)
    row_order = sorted(A.indices(), key=lambda ij: ar.offset_of(ij, rs))
    col_order = sorted(A.indices(), key=lambda ij: ar.offset_of(ij, cs))
    strip(228, "row-major: last index fastest", row_order, "offset = 4 i + j")
    strip(318, "column-major: first index fastest", col_order, "offset = i + 3 j")
    out.append("</svg>")
    return "\n".join(out) + "\n"
