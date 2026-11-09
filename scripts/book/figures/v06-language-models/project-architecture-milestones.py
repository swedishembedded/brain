# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The five milestones of the Volume VI project in build order: the blocks, then the cost model and the parity harness that both
# read them, then the comparison that needs the blocks, the costs and the harness, then the hybrid that reads the comparison.
# Every arrow points from a milestone to one that uses its result; each box names the check that ends the milestone.
#
# Swedish Embedded AB implements solutions for comparing neural network architectures on equal terms. If your team needs
# expertise in architecture evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram  # noqa: E402

W, H = 900, 300
COLS = (110, 340, 570, 790)
ROWS = (60, 150, 240)
BOX_W, BOX_H = 190, 66

MILESTONES = {
    1: (["1  Four blocks", "gradients, chunked = recurrent"], 0, 1, "model"),
    2: (["2  Cost model", "counted = formula"], 1, 0, "model"),
    3: (["3  Parity harness", "tolerance, planted bugs"], 1, 2, "model"),
    4: (["4  Comparison", "paired intervals, verdicts"], 2, 1, "result"),
    5: (["5  Hybrid", "cost against quality"], 3, 1, "model"),
}


def render() -> str:
    d = Diagram(W, H)
    m = {n: d.box(COLS[col], ROWS[row], BOX_W, BOX_H, lines, kind) for n, (lines, col, row, kind) in MILESTONES.items()}
    d.path([m[1]["t"], (m[1]["t"][0], m[2]["l"][1]), m[2]["l"]])
    d.path([m[1]["b"], (m[1]["b"][0], m[3]["l"][1]), m[3]["l"]])
    d.path([m[2]["r"], (m[4]["t"][0], m[2]["r"][1]), m[4]["t"]])
    d.path([m[3]["r"], (m[4]["b"][0], m[3]["r"][1]), m[4]["b"]])
    d.arrow(m[1]["r"], m[4]["l"])
    d.arrow(m[4]["r"], m[5]["l"])
    return d.svg()
