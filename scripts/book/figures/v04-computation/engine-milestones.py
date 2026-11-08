# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The seven milestones of the Volume IV project and which earlier ones each is
# built on. Rows are dependency depth, so no arrow points upward: the tensor
# and oracle (1) are used by every later check, the backend with its naive
# kernel (2) runs everything after it, the tiled kernels (3) feed the reductions
# and epilogue (4) and the parity test (5), and the parity test is the tolerance
# of the quantized product (6) and the gate of the selection (7).
#
# Swedish Embedded AB implements solutions for tensor engines that run one set
# of kernels on every device a client owns. If your team needs expertise in
# compute kernels and backend parity for machine learning inference, you can
# procure our services by sending an email to info@swedishembedded.com.
from svgdiagram import Diagram

W, TOP, PITCH, BOX_W, BOX_H = 800, 50, 96, 220, 54
COLS = (140, 400, 660)

MILESTONES = {
    1: (["1  Tensor and oracle", "shape, strides, float64"], 1, 0),
    2: (["2  CPU backend", "executor, naive kernel"], 1, 1),
    3: (["3  Tiled kernels", "shared memory, registers"], 1, 2),
    4: (["4  Reductions, epilogue", "two passes, fused bias"], 0, 3),
    5: (["5  Parity", "two devices, derived bound"], 1, 3),
    6: (["6  Quantized product", "int8 and int4 groups"], 0, 4),
    7: (["7  Selection", "gate, model clock, choice"], 2, 4),
}


def render() -> str:
    d = Diagram(W, 2 * TOP + 4 * PITCH + 20)
    m = {n: d.box(COLS[col], TOP + layer * PITCH, BOX_W, BOX_H, lines, "result" if n == 7 else "model")
         for n, (lines, col, layer) in MILESTONES.items()}
    d.arrow(m[1]["b"], m[2]["t"])
    d.arrow(m[2]["b"], m[3]["t"])
    d.arrow(m[3]["b"], m[5]["t"])
    d.path([m[3]["l"], (COLS[0], m[3]["l"][1]), m[4]["t"]])
    mid = TOP + 3 * PITCH + PITCH // 2          # halfway between the bottom of row 4 and the top of row 5
    d.path([m[5]["b"], (COLS[1], mid), (COLS[0], mid), m[6]["t"]])
    d.path([m[5]["b"], (COLS[1], mid), (COLS[2], mid), m[7]["t"]])
    return d.svg()
