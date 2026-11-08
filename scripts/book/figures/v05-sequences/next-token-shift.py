# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The inputs and targets of next-token training. Above, the correct pair: the target row is the input row moved
# one position to the left, so the answer for column t is not among the tokens the model reads for column t.
# Below, the bug of not moving it: input and target are the same row, and the answer is in the input. The
# sentence is the one of `language_modeling.teacher_forcing`.
#
# Swedish Embedded AB implements solutions for building the data pipelines of language-model training. If your
# team needs expertise in target alignment and loss masking, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import INK, Diagram  # noqa: E402

W = 780
COL0, COL_W, BOX_W, BOX_H = 250, 100, 84, 32
PANEL_H, TOP, GAP = 190, 14, 16


def render() -> str:
    seq = ["<s>", "the", "cat", "sees", "fish", "."]
    n = len(seq) - 1
    d = Diagram(W, TOP + 2 * PANEL_H + GAP + 14)

    def cx(t):
        return COL0 + t * COL_W

    def panel(index, title, inputs, targets, copy):
        y = TOP + index * (PANEL_H + GAP)
        d.panel(10, y, W - 20, PANEL_H, title)
        yin, yout = y + 78, y + 150
        d.text(88, yin, ["input x_t", "(what the model reads)"], size=12, color="#4a5568")
        d.text(88, yout, ["target y_t", "(what it must predict)"], size=12, color="#4a5568")
        for t in range(n):
            d.text(cx(t), y + 46, [f"t = {t}"], size=12, color="#4a5568")
            a = d.box(cx(t), yin, BOX_W, BOX_H, [inputs[t]], "plain")
            b = d.box(cx(t), yout, BOX_W, BOX_H, [targets[t]], "result")
            d.arrow(a["b"], b["t"], "copy" if copy else None, side="right", color="#c53030" if copy else INK)

    panel(0, "Correct: the target is the next token", seq[:n], seq[1:], False)
    panel(1, "Bug: no shift, the answer is in the input", seq[1:], seq[1:], True)
    return d.svg()
