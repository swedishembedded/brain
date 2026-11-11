# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The toy denoiser of Chapter 11 with its control branch: the frozen base network (grey) with its three skip entries and middle block,
# the trainable copy (blue) with one zero-convolution (yellow) per injection point, and the four places where a residual is added:
# to the skip copies that the up path reads and to the middle activation, never to the running hidden state. The shapes are those of
# `control_and_identity.toy_setup`.
#
# Swedish Embedded AB implements solutions for steering image generators with a pose, an edge map or a face, with every injection
# point checked by name and by size. If your team needs expertise in ControlNet and adapters, you can procure our services by sending
# an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram  # noqa: E402

W, H = 900, 660
DX, UX, CX, PX = 140, 760, 340, 540        # base down column, up column, control copy column, the plus signs
ROWS = (130, 255, 380, 505)
BLUE, ARROW = "accent", "#2b6cb0"


def render() -> str:
    d = Diagram(W, H)
    d.colors.add(ARROW)
    inp = d.oval(DX, 52, 150, 34, ["input 2 x 4"], "plain")
    out = d.oval(UX, 52, 150, 34, ["output 2 x 4"], "plain")
    down = [
        d.box(DX, ROWS[0], 210, 54, ["input convolution", "skip 0, 3 x 4"], "model"),
        d.box(DX, ROWS[1], 210, 54, ["block at 4 positions", "skip 1, 3 x 4"], "model"),
        d.box(DX, ROWS[2], 210, 54, ["block at 2 positions", "skip 2, 4 x 2"], "model"),
        d.box(DX, ROWS[3], 210, 54, ["middle block, site 0", "4 x 2"], "model"),
    ]
    up = [
        d.box(UX, ROWS[0], 210, 54, ["up block joins skip 0", "site 1, output layer"], "model"),
        d.box(UX, ROWS[1], 210, 54, ["up block, doubled", "joins skip 1"], "model"),
        d.box(UX, ROWS[2], 210, 54, ["up block joins", "skip 2 and the middle"], "model"),
    ]
    d.arrow(inp["b"], down[0]["t"])
    d.arrow(up[0]["t"], out["b"])
    d.arrow(down[0]["b"], down[1]["t"])
    d.arrow(down[1]["b"], down[2]["t"], "pool", side="right")
    d.arrow(down[2]["b"], down[3]["t"])
    d.arrow(up[2]["t"], up[1]["b"])
    d.arrow(up[1]["t"], up[0]["b"])

    # the skip lines, each with a plus where the residual joins it
    for k in range(3):
        y = ROWS[k]
        d.arrow(down[k]["r"], up[k]["l"], f"skip {k}", side="above")
        _plus(d, PX, y)
    ym = ROWS[3]
    d.path([down[3]["r"], (UX, ym), up[2]["b"]])
    d.text(PX + 150, ym - 14, ["middle"], size=12, color="#4a5568")
    _plus(d, PX, ym)

    # the control copy: one box and one zero-convolution under each of the four lines
    copies = ["copy of input convolution", "copy of block, 4 positions", "copy of block, 2 positions", "copy of middle block"]
    subs = ["+ embedding of the control image", "", "", "and its site 0"]
    prev = None
    for k in range(4):
        y = ROWS[k] + 62
        c = d.box(CX, y, 220, 42, [copies[k]] + ([subs[k]] if subs[k] else []), BLUE)
        z = d.box(PX, y + 0, 118, 42, [f"zero-conv {k}" if k < 3 else "zero-conv mid"], "result")
        d.arrow(c["r"], z["l"], color=ARROW)
        d.arrow(z["t"], (PX, ROWS[k] + 9), color=ARROW)
        if prev is not None:
            d.arrow(prev["b"], c["t"], color=ARROW)
        prev = c
    d.text(W / 2, 616, ["blue: trained, starts as a copy of the base; yellow: zero-convolutions, start at exactly 0; grey: frozen.",
                        "A residual is added to the skip copy that the up path reads; the running hidden state of the down path is not touched."],
           size=12, color="#4a5568")
    return d.svg()


def _plus(d, x, y):
    d.items.append(f'<circle cx="{x}" cy="{y}" r="9" fill="#ffffff" stroke="#2b6cb0" stroke-width="1.4"/>')
    d.text(x, y, ["+"], size=14, color="#2b6cb0", weight="bold")
