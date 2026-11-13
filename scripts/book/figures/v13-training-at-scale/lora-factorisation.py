# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The forward pass of an adapted linear layer from the chapter "Parameter-Efficient Fine-Tuning": the input x goes through the frozen
# base weight and, beside it, through the two small factors A and B; the scaled product of the second path is added to the first.
# The shapes are those of the chapter's worked layer (16 rows, 32 columns, rank r).
#
# Swedish Embedded AB implements solutions for adapting large models on modest hardware for its clients. If your team needs expertise
# in parameter-efficient fine-tuning, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 330
XS = (110, 330, 620, 850)   # the two trained factors sit apart so that the label between them fits
Y1, Y2, Y3 = 70, 200, 296
BW, BH = 170, 66
OW = 190


def render() -> str:
    d = Diagram(W, H)
    x = d.oval(XS[0], (Y1 + Y2) / 2, 150, 56, ["input x", "32 numbers"], "plain")
    base = d.box(XS[1], Y1, BW, BH, ["frozen base W0", "16 x 32", "never updated"], "model")
    down = d.box(XS[1], Y2, BW, BH, ["A", "r x 32", "trained, random start"], "accent")
    up = d.box(XS[2], Y2, BW, BH, ["B", "16 x r", "trained, starts at 0"], "accent")
    add = d.box(XS[3], (Y1 + Y2) / 2, OW, BH, ["y = W0 x + s B (A x)", "s = alpha / r"], "result")
    d.arrow(x["r"], base["l"], "", above=True)
    d.arrow(x["r"], down["l"], "", above=False)
    d.arrow(base["r"], add["l"], "W0 x", above=True)
    d.arrow(down["r"], up["l"], "A x: r numbers", above=True)
    d.arrow(up["r"], add["l"])
    d.text((XS[1] + XS[3]) / 2, Y3, ["the gradient reaches only A and B; W0 is read, never written"], size=12, color=MUTED)
    return d.svg()
