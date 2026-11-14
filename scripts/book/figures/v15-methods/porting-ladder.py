# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The rungs of a port in the chapter "Porting a Model from Its Reference", in the order they are climbed: facts from the
# references, goldens, the checked mapping, a smoke run at toy size, isolated stages, the chained run, the gradient check, and
# optimization under the frozen gate. Each rung is entered only when the one before it passes.
#
# Swedish Embedded AB implements solutions for porting published models onto its clients' own engines. If your team needs
# expertise in reference goldens and parity testing, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 320
Y1, Y2 = 70, 222
XS = (115, 372, 629, 886)   # four equal columns, from the left
BW, BH = 200, 60


def render() -> str:
    d = Diagram(W, H)
    top = [d.box(XS[0], Y1, BW, BH, ["1  facts", "authority, layout, config"], "plain"),
           d.box(XS[1], Y1, BW, BH, ["2  goldens", "reference, real inputs"], "accent"),
           d.box(XS[2], Y1, BW, BH, ["3  mapping", "coverage and shapes"], "model"),
           d.box(XS[3], Y1, BW, BH, ["4  smoke run", "toy sizes, no weights"], "model")]
    bottom = [d.box(XS[3], Y2, BW, BH, ["5  isolated stages", "golden input to each"], "model"),
              d.box(XS[2], Y2, BW, BH, ["6  chained run", "every tap, end to end"], "model"),
              d.box(XS[1], Y2, BW, BH, ["7  gradient check", "backward vs own loss"], "model"),
              d.box(XS[0], Y2, BW, BH, ["8  optimize", "parity gate frozen"], "result")]
    for a, b in zip(top, top[1:]):
        d.arrow(a["r"], b["l"])
    d.arrow(top[3]["b"], bottom[0]["t"])
    for a, b in zip(bottom, bottom[1:]):
        d.arrow(a["l"], b["r"])
    d.text(W / 2, Y2 + 64, ["a rung is entered only when the one before it passes; a failure is looked for in the newest rung"],
           size=12, color=MUTED)
    return d.svg()
