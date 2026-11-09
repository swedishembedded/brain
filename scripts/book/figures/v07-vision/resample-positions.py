# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Where the outputs of a resize look in the source row, for the half-pixel, corner-aligned and asymmetric conventions, for 4 pixels
# enlarged to 8 and shrunk to 2. Source pixel i spans [i, i + 1] and is sampled at its centre i + 0.5; an output sample at source
# index s is drawn at s + 0.5. Positions are `images.positions_data`, the numbers of the chapter's table.
#
# Swedish Embedded AB implements solutions for matching a model's reference resampling exactly. If your team needs expertise in
# image resampling conventions, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import images as im  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, RED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

W, H = 760, 400
X0, UNIT = 150, 140            # x of source coordinate 0, px per source pixel
ROW = 30                       # px between convention rows
GREEN = "#2f855a"
BLUE = "#2b6cb0"
COLORS = (BLUE, GREEN, RED)
NAMES = ("half-pixel", "corners", "asymmetric")


def _cells(d, y):
    for i in range(4):
        d.items.append(f'<rect x="{fmt(X0 + i * UNIT)}" y="{fmt(y)}" width="{fmt(UNIT)}" height="26" fill="#ebf4ff" '
                       f'stroke="{MUTED}" stroke-width="1.2"/>')
        d.items.append(f'<text x="{fmt(X0 + (i + 0.5) * UNIT)}" y="{fmt(y + 20)}" {FONT} font-size="11" fill="{MUTED}" '
                       f'text-anchor="middle">{i}</text>')


def _panel(d, y, n_in, n_out):
    pos = im.positions_data()[(n_in, n_out)]
    d.text(X0 - 12, y + 13, [f"source, {n_in} pixels"], size=12, anchor="end")
    _cells(d, y)
    for k, (name, ps) in enumerate(zip(NAMES, pos)):
        yy = y + 26 + 22 + k * ROW
        d.text(X0 - 12, yy, [name], size=12, color=COLORS[k], anchor="end")
        d.items.append(f'<line x1="{X0}" y1="{fmt(yy)}" x2="{fmt(X0 + n_in * UNIT)}" y2="{fmt(yy)}" stroke="{FRAME}" stroke-width="1"/>')
        for o, s in enumerate(ps):
            x = X0 + (s + 0.5) * UNIT
            inside = 0 <= s <= n_in - 1
            d.items.append(f'<circle cx="{fmt(x)}" cy="{fmt(yy)}" r="5" fill="{COLORS[k] if inside else "#ffffff"}" stroke="{COLORS[k]}" '
                           f'stroke-width="1.4"/>')
            d.items.append(f'<text x="{fmt(x)}" y="{fmt(yy + 4)}" {FONT} font-size="9" fill="#ffffff" text-anchor="middle">'
                           f'{o if inside else ""}</text>')


def render() -> str:
    d = Diagram(W, H)
    d.text(W / 2, 18, ["Output samples in the source row (dot i = output pixel i; hollow = outside the source, clamped)"], size=13)
    d.panel(20, 34, W - 40, 160, "4 pixels enlarged to 8")
    _panel(d, 70, 4, 8)
    d.panel(20, 210, W - 40, 160, "4 pixels shrunk to 2")
    _panel(d, 246, 4, 2)
    return d.svg()
