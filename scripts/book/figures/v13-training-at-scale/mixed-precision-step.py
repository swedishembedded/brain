# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One step of mixed-precision training as the chapter "Mixed-Precision Training" builds it: the fp32 master is rounded to a narrow
# working copy, the forward and backward passes run on that copy with the loss multiplied by S, the scaled gradients are divided by S
# and tested for overflow in fp32, and the optimizer updates the master, never the working copy. A step with a non-finite gradient is
# skipped and the scale is halved.
#
# Swedish Embedded AB implements solutions for memory-bounded low-precision training for its clients. If your team needs expertise in
# training large models in 16 and 8 bits, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 360
Y1, Y2, Y3 = 62, 200, 316
XS = (110, 340, 570, 800)   # four equal columns, from the left
BW, BH = 170, 62


def render() -> str:
    d = Diagram(W, H)
    master = d.box(XS[0], (Y1 + Y2) / 2, BW, 190, ["master", "fp32 value", "", "moments", "fp32 or 8-bit"], "accent")
    work = d.box(XS[1], Y1, BW, BH, ["working copy", "round(master)", "bf16 or fp8"], "model")
    fwd = d.box(XS[2], Y1, BW, BH, ["forward, backward", "on the loss S x L", "narrow products"], "model")
    grad = d.box(XS[3], Y1, BW, BH, ["scaled gradient", "S x g", "narrow format"], "model")
    unscale = d.box(XS[3], Y2, BW, BH, ["unscale", "g = (S x g) / S", "fp32, finite?"], "accent")
    clip = d.box(XS[2], Y2, BW, BH, ["clip", "by the global norm", "fp32"], "model")
    adam = d.box(XS[1], Y2, BW, BH, ["optimizer rule", "on the master", "fp32 arithmetic"], "accent")
    skip = d.oval(XS[3], Y3, BW, 44, ["not finite: skip the step", "S = S / 2"], "result")
    d.arrow((master["r"][0], Y1), work["l"], "cast down", above=True)
    d.arrow(work["r"], fwd["l"], "weights")
    d.arrow(fwd["r"], grad["l"], "backward")
    d.arrow(grad["b"], unscale["t"], "", side="right")
    d.arrow(unscale["l"], clip["r"], "finite", above=True)
    d.arrow(clip["l"], adam["r"], "g", above=True)
    d.arrow(adam["l"], (master["r"][0], Y2))
    d.arrow(unscale["b"], skip["t"], "", side="right", dash="5 3")
    d.text((XS[0] + XS[2]) / 2 - 20, Y3 + 6, ["every update lands on the master;", "the working copy is rebuilt from it"], size=12, color=MUTED)
    return d.svg()
