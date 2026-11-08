# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The weights of the additive attention of the trained model (run 0) on one input, as a grid: one row per output step, one
# column per input digit, darker where the weight is larger. The cell a reversal would read at each step is outlined.
# Numbers from `seq2seq.alignment_data`, the same as the table of the chapter.
#
# Swedish Embedded AB implements solutions for models that turn one sequence into another. If your team needs expertise in
# attention and alignment in sequence models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import seq2seq as s2s  # noqa: E402
from svgdiagram import INK, MUTED, RED, Diagram  # noqa: E402
from svgplot import fmt  # noqa: E402

CELL_W, CELL_H = 70, 38
X0, Y0 = 150, 90


def _shade(w):
    """White at 0 to the book's blue at 1."""
    lo, hi = (255, 255, 255), (43, 108, 176)
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * w) for a, b in zip(lo, hi))


def render() -> str:
    src, out, w = s2s.alignment_data()
    T, rows = len(src), len(w)
    width = X0 + T * CELL_W + 60
    height = Y0 + rows * CELL_H + 70
    d = Diagram(width, height)
    d.text(X0 + T * CELL_W / 2, 24, ["input digits, in the order they are read"], size=13, color=MUTED)
    for i in range(T):
        d.text(X0 + (i + 0.5) * CELL_W, 62, [f"x{i + 1} = {src[i]}"], size=13)
    d.text(70, Y0 - 22, ["output step j", "(token written)"], size=12, color=MUTED)
    for j in range(rows):
        cy = Y0 + (j + 0.5) * CELL_H
        d.text(80, cy, [f"{j + 1}   ({s2s.TOKEN_NAMES[out[j]]})"], size=13, anchor="middle")
        for i in range(T):
            x, wt = X0 + i * CELL_W, w[j][i]
            d.items.append(f'<rect x="{fmt(x)}" y="{fmt(Y0 + j * CELL_H)}" width="{CELL_W}" height="{CELL_H}" fill="{_shade(wt)}" '
                           f'stroke="#ffffff" stroke-width="1"/>')
            d.text(x + CELL_W / 2, cy, [f"{wt:.2f}"], size=12, color="#ffffff" if wt > 0.45 else INK)
        if j < T:
            i = T - 1 - j       # the input a reversal reads at step j
            d.items.append(f'<rect x="{fmt(X0 + i * CELL_W + 2)}" y="{fmt(Y0 + j * CELL_H + 2)}" width="{CELL_W - 4}" '
                           f'height="{CELL_H - 4}" fill="none" stroke="{RED}" stroke-width="2"/>')
    d.text(X0 + T * CELL_W / 2, Y0 + rows * CELL_H + 30,
           ["darker = larger weight, each row adds to 1; red outline = the digit a reversal copies at that step"], size=12, color=MUTED)
    return d.svg()
