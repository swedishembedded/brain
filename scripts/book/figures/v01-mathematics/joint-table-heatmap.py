# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Heat map of a joint distribution with its marginals.

The joint table of a three-class classifier (true class by predicted class)
comes from the chapter's own counts; the marginals are the row and column sums
computed here, so the picture and the table beside it cannot disagree.

Swedish Embedded AB implements solutions for explaining classifier behaviour to
engineering teams. If your team needs expertise in evaluating probabilistic
models, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probability as pr  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

CELL = 84


def _shade(v, vmax):
    """White to blue by probability (square-root scale so small cells stay visible)."""
    t = (v / vmax) ** 0.5
    lo, hi = (237, 242, 247), (43, 108, 176)
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(lo, hi)), t


def _text(x, y, s, size=13, anchor="middle", color="#1a202c", weight="normal"):
    return (f'<text x="{fmt(x)}" y="{fmt(y)}" {FONT} font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')


def render() -> str:
    m = [[float(v) for v in row] for row in pr.classifier_joint_matrix()]
    n = len(pr.CLASSES)
    ox, oy = 96, 62
    w, h = ox + n * CELL + CELL + 20, oy + n * CELL + CELL + 28
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
           _text(ox + n * CELL / 2, 20, "predicted class", 13, weight="bold"),
           _text(14, oy + n * CELL / 2, "true class", 13, weight="bold").replace(
               "<text ", f'<text transform="rotate(-90 14 {fmt(oy + n * CELL / 2)})" ', 1)]
    vmax = max(max(r) for r in m)
    for j, name in enumerate(pr.CLASSES):
        out.append(_text(ox + j * CELL + CELL / 2, oy - 10, name))
    out.append(_text(ox + n * CELL + CELL / 2, oy - 10, "P(true)", 12, color="#c05621", weight="bold"))
    for i, name in enumerate(pr.CLASSES):
        out.append(_text(ox - 8, oy + i * CELL + CELL / 2 + 4, name, anchor="end"))
        for j in range(n):
            fill, t = _shade(m[i][j], vmax)
            out.append(f'<rect x="{ox + j * CELL}" y="{oy + i * CELL}" width="{CELL}" height="{CELL}" '
                       f'fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            out.append(_text(ox + j * CELL + CELL / 2, oy + i * CELL + CELL / 2 + 5, f"{m[i][j]:.3f}",
                             15, color="#ffffff" if t > 0.55 else "#1a202c"))
        row = sum(m[i])
        out.append(f'<rect x="{ox + n * CELL}" y="{oy + i * CELL}" width="{CELL}" height="{CELL}" fill="#fffaf0" '
                   'stroke="#ffffff" stroke-width="2"/>')
        out.append(_text(ox + n * CELL + CELL / 2, oy + i * CELL + CELL / 2 + 5, f"{row:.3f}", 15, color="#c05621", weight="bold"))
    out.append(_text(ox - 8, oy + n * CELL + CELL / 2 + 4, "P(predicted)", 12, anchor="end", color="#c05621", weight="bold"))
    for j in range(n):
        col = sum(m[i][j] for i in range(n))
        out.append(f'<rect x="{ox + j * CELL}" y="{oy + n * CELL}" width="{CELL}" height="{CELL}" fill="#fffaf0" '
                   'stroke="#ffffff" stroke-width="2"/>')
        out.append(_text(ox + j * CELL + CELL / 2, oy + n * CELL + CELL / 2 + 5, f"{col:.3f}", 15, color="#c05621", weight="bold"))
    total = sum(map(sum, m))
    out.append(f'<rect x="{ox + n * CELL}" y="{oy + n * CELL}" width="{CELL}" height="{CELL}" fill="#fffaf0" '
               'stroke="#ffffff" stroke-width="2"/>')
    out.append(_text(ox + n * CELL + CELL / 2, oy + n * CELL + CELL / 2 + 5, f"{total:.3f}", 15, color="#c05621", weight="bold"))
    out.append(_text(ox, h - 8, "darker = more probability; orange = marginals (sums)", 11, anchor="start", color="#4a5568"))
    out.append("</svg>")
    return "\n".join(out) + "\n"
