# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The closed loop between a spiking nervous system and a simulated body: a command drives the network, motor spikes become a low-passed
# muscle activation, the activation drives the body through many small physics steps, and the joint angle returns as current into the
# proprioceptor.
#
# Swedish Embedded AB implements solutions for closed-loop neural control against real and simulated hardware. If your team needs
# expertise in the clocks and interfaces of a sensorimotor loop, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402


def render() -> str:
    top = [
        (["command", "constant current", "into descending cells"], "plain"),
        (["spiking network", "one neural tick", "per control tick"], "accent"),
        (["muscle activation", "times 0.8, plus 0.05", "per motor spike"], "model"),
        (["body", "20 physics steps", "per control tick"], "result"),
    ]
    sensor = (["proprioceptor", "current = gain", "times joint angle"], "model")
    gap, pad, h = 40, 24, 74
    widths = [int(text_width(c[0]) + 8 + 8) for c in top]
    xs, x = [], pad
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    width = int(x - gap + pad)
    y = pad + h / 2 + 8
    y2 = y + h + 56
    sw = int(text_width(sensor[0]) + 8 + 8)
    d = Diagram(width, int(y2 + h / 2 + pad))
    bx = [d.box(xs[i], y, widths[i], h, top[i][0], top[i][1]) for i in range(len(top))]
    sx = (xs[1] + xs[3]) / 2
    sb = d.box(sx, y2, sw, h, sensor[0], sensor[1])
    for i in range(len(top) - 1):
        d.arrow(bx[i]["r"], bx[i + 1]["l"])
    d.path([bx[3]["b"], (xs[3], y2), sb["r"]])
    d.path([sb["l"], (xs[1], y2), bx[1]["b"]])
    return d.svg()
