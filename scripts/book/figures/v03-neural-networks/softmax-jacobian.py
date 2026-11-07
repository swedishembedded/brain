# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The Jacobian diag(p) - p p^T of the softmax at the chapter's part, as a heat map with its values.

Swedish Embedded AB implements solutions for deriving and verifying the gradients of classification losses. If
your team needs expertise in backpropagation through softmax layers, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import softmax  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL, LEFT, TOP = 78, 118, 78
INK, MUTED = "#1a202c", "#4a5568"
POS, NEG = (192, 86, 33), (43, 108, 176)  # orange for positive entries, blue for negative ones


def _shade(v, vmax):
    """White at 0, the full colour at +-vmax."""
    base = POS if v >= 0 else NEG
    k = min(abs(v) / vmax, 1.0)
    return "#" + "".join(f"{round(255 + (c - 255) * k):02x}" for c in base)


def render() -> str:
    p = softmax._softmax(softmax._logits(softmax.X))
    J = softmax._jacobian(p)
    n = softmax.C
    vmax = max(abs(v) for row in J for v in row)
    W, H = LEFT + n * CELL + 12, TOP + n * CELL + 12
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{fmt(LEFT + n * CELL / 2)}" y="20" {FONT} font-size="13" font-weight="bold" fill="{INK}" '
           'text-anchor="middle">Jacobian of the softmax, entry (i, j) = d p[i] / d z[j]</text>',
           f'<text x="{fmt(LEFT + n * CELL / 2)}" y="38" {FONT} font-size="11" fill="{MUTED}" text-anchor="middle">'
           'orange: raising z[j] raises p[i]; blue: it lowers p[i]</text>']
    for j in range(n):
        out.append(f'<text x="{fmt(LEFT + j * CELL + CELL / 2)}" y="{TOP - 12}" {FONT} font-size="12" fill="{INK}" '
                   f'text-anchor="middle">z[{j}] {escape(softmax.CLASSES[j])}</text>')
    for i in range(n):
        out.append(f'<text x="{LEFT - 10}" y="{fmt(TOP + i * CELL + CELL / 2 + 4)}" {FONT} font-size="12" fill="{INK}" '
                   f'text-anchor="end">p[{i}] {escape(softmax.CLASSES[i])}</text>')
        for j in range(n):
            v = J[i][j]
            x, y = LEFT + j * CELL, TOP + i * CELL
            out.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="{_shade(v, vmax)}" stroke="#ffffff" '
                       'stroke-width="2"/>')
            out.append(f'<text x="{fmt(x + CELL / 2)}" y="{fmt(y + CELL / 2 + 4)}" {FONT} font-size="13" fill="{INK}" '
                       f'text-anchor="middle">{v:.4f}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
