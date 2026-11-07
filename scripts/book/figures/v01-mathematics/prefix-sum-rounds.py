# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The doubling prefix sum of eight numbers: which cell each new value is built from.

Row 0 is the input; row r holds the array after round r. In round r every
cell i at least 2^(r-1) from the left edge adds the cell 2^(r-1) places to
its left (the arrow); the other cells copy themselves down. The values are
taken from the worked module `algorithms`, so the figure shows the table.

Swedish Embedded AB implements solutions for data-parallel primitives on
GPUs. If your team needs expertise in scans, sorts and reductions for
accelerators, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import algorithms  # noqa: E402
from svgplot import FONT, PALETTE  # noqa: E402

CELL, ROW, LEFT, TOP = 62, 74, 120, 44


def render() -> str:
    x = algorithms.SCAN_DEMO
    n = len(x)
    rounds = algorithms._doubling_rounds(x)
    grid = [("input", x)] + [(f"after offset {o}", arr) for o, arr, _ in rounds]
    width, height = LEFT + n * CELL + 20, TOP + len(grid) * ROW
    cx = lambda i: LEFT + i * CELL + CELL / 2  # noqa: E731
    cy = lambda r: TOP + r * ROW  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
           '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
           'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#4a5568"/></marker></defs>',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    for i in range(n):
        out.append(f'<text x="{cx(i)}" y="22" {FONT} font-size="12" fill="#4a5568" text-anchor="middle">i = {i}</text>')
    for r, (label, arr) in enumerate(grid):
        y = cy(r)
        out.append(f'<text x="{LEFT - 10}" y="{y + 26}" {FONT} font-size="12" fill="#1a202c" text-anchor="end">{escape(label)}</text>')
        for i, v in enumerate(arr):
            changed = r > 0 and v != grid[r - 1][1][i]
            fill = "#fefcbf" if changed else "#e2e8f0"
            out.append(f'<rect x="{LEFT + i * CELL + 6}" y="{y + 8}" width="{CELL - 12}" height="28" rx="4" '
                       f'fill="{fill}" stroke="#4a5568"/>')
            out.append(f'<text x="{cx(i)}" y="{y + 27}" {FONT} font-size="13" fill="#1a202c" text-anchor="middle">{v}</text>')
        if r > 0:
            offset = rounds[r - 1][0]
            color = PALETTE[(r - 1) % len(PALETTE)]
            for i in range(offset, n):
                out.append(f'<line x1="{cx(i - offset)}" y1="{cy(r - 1) + 36}" x2="{cx(i) - 3}" y2="{y + 6}" '
                           f'stroke="{color}" stroke-width="1.4" marker-end="url(#head)"/>')
            for i in range(n):             # every cell keeps its own old value as one operand
                out.append(f'<line x1="{cx(i)}" y1="{cy(r - 1) + 36}" x2="{cx(i)}" y2="{y + 6}" '
                           f'stroke="#a0aec0" stroke-width="1"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
