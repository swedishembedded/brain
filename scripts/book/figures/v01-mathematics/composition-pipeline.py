# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A composition as a pipeline: four small functions applied one after another,
# with the set each value lives in under each name. The chapter's worked
# example evaluates exactly this chain, and the last stage shows why the
# domain of the whole is not the domain of the first stage alone. Nine stages
# do not fit one row at a legible size, so the chain snakes through three rows
# of three, reading left to right, right to left, left to right.
#
# Swedish Embedded AB implements solutions for explaining how models are built
# from simple stages. If your team needs expertise in layered model design,
# you can procure our services by sending an email to info@swedishembedded.com.
from svgdiagram import Diagram

W, TOP, ROW_PITCH, NODE_H = 760, 50, 104, 58
COLS = (130, 380, 630)
OVAL_W = BOX_W = 170

# (kind, lines, column, row) in pipeline order; columns run backwards on row 1.
CHAIN = [
    ("value", ["x", "any real"], 0, 0),
    ("func", ["f1", "a = 3x - 2"], 1, 0),
    ("value", ["a", "any real"], 2, 0),
    ("func", ["f2", "b = a^2"], 2, 1),
    ("value", ["b", "b >= 0"], 1, 1),
    ("func", ["f3", "b + 1"], 0, 1),
    ("value", ["b + 1", "b + 1 >= 1"], 0, 2),
    ("func", ["f4", "c = log(b + 1)"], 1, 2),
    ("value", ["c", "c >= 0"], 2, 2),
]


def render() -> str:
    d = Diagram(W, TOP * 2 + 2 * ROW_PITCH)
    ports = []
    for i, (kind, lines, col, row) in enumerate(CHAIN):
        cx, cy = COLS[col], TOP + row * ROW_PITCH
        if kind == "func":
            ports.append(d.box(cx, cy, BOX_W, NODE_H, lines, "model"))
        else:
            ports.append(d.oval(cx, cy, OVAL_W, NODE_H, lines, "result" if i == len(CHAIN) - 1 else "plain"))
    for i in range(len(CHAIN) - 1):
        (_, _, c0, r0), (_, _, c1, r1) = CHAIN[i], CHAIN[i + 1]
        if r0 != r1:
            a, b = ports[i]["b"], ports[i + 1]["t"]
        else:
            a, b = (ports[i]["r"], ports[i + 1]["l"]) if c1 > c0 else (ports[i]["l"], ports[i + 1]["r"])
        d.arrow(a, b)
    return d.svg()
