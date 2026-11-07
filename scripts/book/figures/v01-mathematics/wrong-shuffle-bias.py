# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Exact probabilities of the six orders of three items under two shuffles.

Both sets of bars come from enumerating every equally likely sequence of random
choices (27 for the wrong shuffle, 6 for Fisher-Yates), not from sampling. The
dashed line is the uniform value 1/6.

Swedish Embedded AB implements solutions for unbiased data ordering in training
pipelines. If your team needs expertise in sampling and shuffling for machine
learning data you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import randomness as rn  # noqa: E402
from svgplot import FONT, PALETTE, fmt  # noqa: E402

W, H = 560, 330
LEFT, RIGHT, TOP, BOTTOM = 58, 18, 40, 56
YMAX = 0.21


def render() -> str:
    wrong = rn.wrong_shuffle_probabilities(3)
    fy = rn._fisher_yates_counts(3)
    n = len(wrong)
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM
    X = lambda v: LEFT + v / n * pw  # noqa: E731
    Y = lambda v: TOP + ph - v / YMAX * ph  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for t in (0.0, 0.05, 0.10, 0.15, 0.20):
        out.append(f'<line x1="{LEFT}" y1="{fmt(Y(t))}" x2="{W - RIGHT}" y2="{fmt(Y(t))}" stroke="#edf2f7"/>')
        out.append(f'<text x="{LEFT - 6}" y="{fmt(Y(t) + 4)}" {FONT} font-size="11" fill="#4a5568" text-anchor="end">{t:.2f}</text>')
    bw = 0.36
    for i, (perm, pr) in enumerate(wrong):
        for off, val, col in ((0.1, float(pr), PALETTE[3]), (0.54, float(Fraction(fy[perm], 6)), PALETTE[0])):
            out.append(f'<rect x="{fmt(X(i + off))}" y="{fmt(Y(val))}" width="{fmt(X(i + off + bw) - X(i + off))}" '
                       f'height="{fmt(Y(0) - Y(val))}" fill="{col}"/>')
        out.append(f'<text x="{fmt(X(i + 0.28))}" y="{fmt(Y(float(pr)) - 5)}" {FONT} font-size="11" fill="{PALETTE[3]}" '
                   f'text-anchor="middle">{int(pr * 27)}/27</text>')
        out.append(f'<text x="{fmt(X(i + 0.5))}" y="{fmt(Y(0) + 17)}" {FONT} font-size="12" fill="#1a202c" '
                   f'text-anchor="middle">{" ".join(map(str, perm))}</text>')
    out.append(f'<line x1="{LEFT}" y1="{fmt(Y(1 / 6))}" x2="{W - RIGHT}" y2="{fmt(Y(1 / 6))}" stroke="#4a5568" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{W - RIGHT - 4}" y="{fmt(Y(1 / 6) - 5)}" {FONT} font-size="11" fill="#4a5568" text-anchor="end">uniform: 1/6</text>')
    out.append(f'<line x1="{LEFT}" y1="{fmt(Y(0))}" x2="{W - RIGHT}" y2="{fmt(Y(0))}" stroke="#718096"/>')
    out.append(f'<text x="{fmt(LEFT + pw / 2)}" y="{H - 12}" {FONT} font-size="12" fill="#1a202c" text-anchor="middle">'
               'resulting order of the items 0 1 2</text>')
    out.append(f'<text x="14" y="{fmt(TOP + ph / 2)}" {FONT} font-size="12" fill="#1a202c" text-anchor="middle" '
               f'transform="rotate(-90 14 {fmt(TOP + ph / 2)})">probability</text>')
    out.append(f'<text x="{W / 2}" y="18" {FONT} font-size="13" font-weight="bold" fill="#1a202c" text-anchor="middle">'
               'Probability of each order after three items are shuffled</text>')
    for k, (label, col) in enumerate((("wrong shuffle (27 paths)", PALETTE[3]), ("Fisher-Yates (6 paths)", PALETTE[0]))):
        lx = LEFT + 8 + k * 190
        out.append(f'<rect x="{lx}" y="26" width="10" height="10" fill="{col}"/>')
        out.append(f'<text x="{lx + 15}" y="35" {FONT} font-size="11" fill="#1a202c">{label}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
