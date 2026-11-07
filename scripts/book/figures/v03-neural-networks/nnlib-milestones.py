# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The seven milestones of the Volume III project and which earlier ones each
# builds on. Milestones 1 (layers), 3 (optimizers, schedule, clip) and 4
# (initialisation) stand on Volume I alone; the checker (2) needs the layers,
# the training loop (5) needs layers, optimizers and initialisation, the two
# models (6) need the loop and the checker, and the debugging exercise (7) runs
# on the classifier of milestone 6. Rows are dependency depth, so no arrow
# points upward.
#
# Swedish Embedded AB implements solutions for neural network libraries whose
# gradients are proven. If your team needs expertise in building and verifying
# training code, you can procure our services by sending an email to
# info@swedishembedded.com.
from svgdiagram import Diagram

W, TOP, PITCH, BOX_W, BOX_H = 800, 50, 96, 200, 54
COLS = (140, 400, 660)

MILESTONES = {
    1: (["1  Layers", "forward, backward, saved state"], 0, 0),
    3: (["3  Optimizers", "SGD to Lion, schedule, clip"], 1, 0),
    4: (["4  Initialisation", "He, Xavier, fan-in"], 2, 0),
    2: (["2  Gradient checker", "per entry, directional"], 0, 1),
    5: (["5  Training loop", "log, stop, checkpoint, resume"], 1, 1),
    6: (["6  Two models", "classifier and autoencoder"], 1, 2),
    7: (["7  Debugging", "three planted defects"], 1, 3),
}


def render() -> str:
    d = Diagram(W, 2 * TOP + 3 * PITCH + 20)
    m = {n: d.box(COLS[col], TOP + layer * PITCH, BOX_W, BOX_H, lines, "result" if n == 7 else "model")
         for n, (lines, col, layer) in MILESTONES.items()}

    def at(box, edge, dx):
        return (box[edge][0] + dx, box[edge][1])

    mid = TOP + PITCH // 2 + BOX_H // 4
    d.arrow(m[1]["b"], m[2]["t"])
    d.arrow(m[3]["b"], m[5]["t"])
    d.path([at(m[1], "b", 60), (COLS[0] + 60, mid), (COLS[1] - 60, mid), at(m[5], "t", -60)])
    d.path([m[4]["b"], (COLS[2], m[5]["l"][1]), m[5]["r"]])
    d.arrow(m[5]["b"], m[6]["t"])
    d.path([m[2]["b"], (COLS[0], m[6]["l"][1]), m[6]["l"]])
    d.arrow(m[6]["b"], m[7]["t"])
    return d.svg()
