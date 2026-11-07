# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Broadcasting [3, 1] + [1, 4] -> [3, 4]: each operand is stretched along its size-1 axis.

The stretched copies are drawn as faded cells; they are not stored, the view
reads the same number again (stride 0). The sum is computed with the chapter's
worked module.

Swedish Embedded AB implements solutions for correct shape handling in tensor
code. If your team needs expertise in broadcasting semantics and their cost,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
import arrays as ar  # noqa: E402

FONT = 'font-family="Helvetica, Arial, sans-serif"'
CW = 40


def text(x, y, s, size=12, anchor="start", bold=False, color="#1a202c"):
    wt = ' font-weight="bold"' if bold else ""
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{color}" text-anchor="{anchor}"{wt}>{escape(s)}</text>'


def grid(x0, y0, view, fill, stroke, stored_shape):
    """Cells of `view` (a broadcast view); cells outside the stored extent are faded."""
    out = []
    for i, j in view.indices():
        stored = i < stored_shape[0] and j < stored_shape[1]
        x, y = x0 + j * CW, y0 + i * CW
        dash = "" if stored else ' stroke-dasharray="3 3" opacity="0.55"'
        out.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CW}" fill="{fill}" stroke="{stroke}"{dash}/>')
        out.append(text(x + CW / 2, y + CW / 2 + 4.5, str(view.at((i, j))), 13, "middle",
                        color="#1a202c" if stored else "#718096"))
    return out


def render() -> str:
    col = ar.from_nested([[10], [20], [30]])        # [3, 1]
    row = ar.from_nested([[1, 2, 3, 4]])            # [1, 4]
    shape = ar.broadcast_shape(col.shape, row.shape)
    total = ar.add(col, row)
    cv, rv = col.broadcast_to(shape), row.broadcast_to(shape)
    gap = 60
    w1 = 4 * CW
    width = 3 * w1 + 2 * gap + 40
    height = 3 * CW + 110
    y0 = 60
    x1, x2, x3 = 20, 20 + w1 + gap, 20 + 2 * (w1 + gap)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    out.append(text(x1, 24, "a: shape [3, 1]", 13, bold=True))
    out.append(text(x1, 42, "stretched to [3, 4], strides (1, 0)", 11, color="#4a5568"))
    out += grid(x1, y0, cv, "#bee3f8", "#2b6cb0", col.shape)
    out.append(text(x2 - gap / 2, y0 + 1.5 * CW + 6, "+", 22, "middle", bold=True))
    out.append(text(x2, 24, "b: shape [1, 4]", 13, bold=True))
    out.append(text(x2, 42, "stretched to [3, 4], strides (0, 1)", 11, color="#4a5568"))
    out += grid(x2, y0, rv, "#fed7aa", "#c05621", row.shape)
    out.append(text(x3 - gap / 2, y0 + 1.5 * CW + 6, "=", 22, "middle", bold=True))
    out.append(text(x3, 24, "a + b: shape [3, 4]", 13, bold=True))
    out.append(text(x3, 42, "one add per output element", 11, color="#4a5568"))
    out += grid(x3, y0, total, "#c6f6d5", "#2f855a", total.shape)
    out.append(text(20, height - 14, "Solid cells are stored numbers; faded dashed cells are the same number read again (stride 0).",
                    11, color="#4a5568"))
    out.append("</svg>")
    return "\n".join(out) + "\n"
