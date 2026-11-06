# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The real number line with integers, fractions and irrational numbers marked at their computed positions.

Swedish Embedded AB implements solutions for explaining the number systems
behind machine learning. If your team needs expertise in numerical methods,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numbasics  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

W, H = 640, 190
X0, X1 = 40, 610
LO, HI = -2.5, 4.5
Y = 95
COLORS = {"integer": "#2b6cb0", "rational": "#2f855a", "irrational": "#c05621"}
# Integers are labelled below the line, fractions and irrational numbers above,
# in separate rows so that neighbours such as 3/2 and sqrt 2 do not collide.
LABEL_Y = {"integer": Y + 26, "rational": Y - 18, "irrational": Y - 46}


def px(v):
    return X0 + (v - LO) / (HI - LO) * (X1 - X0)


def render() -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<line x1="{X0 - 12}" y1="{Y}" x2="{X1 + 12}" y2="{Y}" stroke="#4a5568" stroke-width="2"/>']
    for n in range(-2, 5):
        out.append(f'<line x1="{fmt(px(n))}" y1="{Y - 6}" x2="{fmt(px(n))}" y2="{Y + 6}" stroke="#4a5568"/>')
    for label, v, kind in numbasics.number_line_points():
        c, x, ty = COLORS[kind], px(v), LABEL_Y[kind]
        out.append(f'<circle cx="{fmt(x)}" cy="{Y}" r="5" fill="{c}"/>')
        if kind != "integer":
            out.append(f'<line x1="{fmt(x)}" y1="{Y - 7}" x2="{fmt(x)}" y2="{ty + 4}" stroke="{c}" stroke-dasharray="2 2"/>')
        out.append(f'<text x="{fmt(x)}" y="{ty}" {FONT} font-size="13" fill="{c}" text-anchor="middle">{escape(label)}</text>')
    lx = 40
    for kind, text in (("integer", "integers"), ("rational", "fractions (rational numbers)"), ("irrational", "irrational numbers")):
        out.append(f'<circle cx="{lx}" cy="{H - 18}" r="5" fill="{COLORS[kind]}"/>')
        out.append(f'<text x="{lx + 10}" y="{H - 14}" {FONT} font-size="12" fill="#1a202c">{text}</text>')
        lx += 20 + 8 * len(text) + 40
    out.append("</svg>")
    return "\n".join(out) + "\n"
