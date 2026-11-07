# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Transposing without moving data: two readings of the same twelve numbers.

Both panels draw the storage of A (shape [3, 4], row-major, strides (4, 1)) as
the grid it is usually drawn as. The number in each cell is the position at
which a row-major reading of the array visits that cell. Left: A itself, which
walks the storage in order. Right: the view B = A^T of shape [4, 3] with
strides (1, 4), which visits the same cells but jumps down the columns. The
visit order is computed from the strides by the chapter's worked module.

Swedish Embedded AB implements solutions for avoiding needless copies in
tensor pipelines. If your team needs expertise in strides and memory traffic,
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
CW = 52


def text(x, y, s, size=12, anchor="start", bold=False, color="#1a202c"):
    wt = ' font-weight="bold"' if bold else ""
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{color}" text-anchor="{anchor}"{wt}>{escape(s)}</text>'


def panel(x0, y0, title, subtitle, view, A):
    """Draw A's storage grid; each cell shows the step at which `view` reads it."""
    out = [text(x0, y0 - 30, title, 13, bold=True), text(x0, y0 - 12, subtitle, 12, color="#4a5568")]
    step_of = {}
    for step, idx in enumerate(view.indices()):
        off = ar.offset_of(idx, view.strides, view.offset)
        step_of[divmod(off, A.shape[1])] = step     # storage position -> (row of A, column of A)
    centres = {}
    for (i, j), step in step_of.items():
        x, y = x0 + j * CW, y0 + i * CW
        out.append(f'<rect x="{x}" y="{y}" width="{CW}" height="{CW}" fill="{FILLS[i]}" stroke="{STROKES[i]}"/>')
        centres[step] = (x + CW / 2, y + CW / 2)
    pts = " ".join(f"{centres[s][0]:.1f},{centres[s][1]:.1f}" for s in sorted(centres))
    out.append(f'<polyline fill="none" stroke="#4a5568" stroke-width="1.4" stroke-dasharray="4 3" points="{pts}"/>')
    for step, (cx, cy) in centres.items():
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="#ffffff" stroke="#4a5568"/>')
        out.append(text(f"{cx:.1f}", f"{cy + 4.5:.1f}", str(step), 12, "middle"))
    return out


def render() -> str:
    A = ar.Arr(list(range(12)), (3, 4))
    B = A.permute((1, 0))
    width, height = 2 * 4 * CW + 100, 3 * CW + 120
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    out += panel(20, 70, "A: shape [3, 4], strides (4, 1)", "reads the storage in order", A, A)
    out += panel(60 + 4 * CW, 70, "B = A^T: shape [4, 3], strides (1, 4)", "same storage, jumps of 4", B, A)
    out.append(text(20, height - 14, "Cell colour = storage row of A. Number = step at which a row-major reading visits the cell.",
                    11, color="#4a5568"))
    out.append("</svg>")
    return "\n".join(out) + "\n"
