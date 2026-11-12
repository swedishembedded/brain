# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The loop of an interactive world model: keys become one action, the model turns its context and that action into the next frame, the
# frame is shown and also becomes context for the step after. A reset rewinds the context to the frame the session started from.
#
# Swedish Embedded AB implements solutions for real-time interactive generative models. If your team needs expertise in playable world
# models and their input and display loops, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402


def render() -> str:
    cols = [
        (["keys held", "(a set)"], "plain"),
        (["chord map", "longest chord", "wins: one action"], "model"),
        (["world model", "context + action", "-> next frame"], "accent"),
        (["frame", "C x H x W"], "result"),
        (["window or", "recorder"], "plain"),
    ]
    gap, pad, h = 36, 24, 74
    widths = [int(text_width(c[0]) + 8 + 8) for c in cols]
    xs, x = [], pad
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    width = int(x - gap + pad)
    y = pad + h / 2 + 30
    d = Diagram(width, int(y + h / 2 + 70 + pad))
    bx = [d.box(xs[i], y, widths[i], h, cols[i][0], cols[i][1]) for i in range(len(cols))]
    for i in range(len(cols) - 1):
        d.arrow(bx[i]["r"], bx[i + 1]["l"])
    low = y + h / 2 + 44
    d.path([bx[3]["b"], (xs[3], low), (xs[2], low), bx[2]["b"]])
    d.text((xs[2] + xs[3]) / 2, low + 16, ["the frame joins the context"], size=12, color="#4a5568")
    top = y - h / 2 - 22
    d.path([(xs[1], y - h / 2), (xs[1], top), (xs[2] - 30, top), (xs[2] - 30, y - h / 2)], dash="4 3")
    d.text((xs[1] + xs[2] - 30) / 2, top - 12, ["reset: restore the first context"], size=12, color="#4a5568")
    return d.svg()
