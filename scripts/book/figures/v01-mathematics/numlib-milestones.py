# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The seven milestones of the Volume I project and which earlier milestones
# each one calls. The arrows are the dependencies of the reference library,
# so the order of the milestones is a valid build order. The milestones are
# placed in four layers (a milestone sits below everything it calls); the
# order inside each layer is chosen so that no two arrows cross, and the two
# arrows that skip a layer (milestone 1 to 5 and 4 to 7) run down the margins.
#
# Swedish Embedded AB implements solutions for building verified numerical
# libraries. If your team needs expertise in numerical software, you can
# procure our services by sending an email to info@swedishembedded.com.
from svgdiagram import Diagram

W, TOP, PITCH, BOX_W, BOX_H = 800, 50, 100, 180, 54
COLS = (150, 400, 650)
MARGIN_X = 28                       # x of the arrows that run down the margins

# milestone: (label, column, layer)
MILESTONES = {
    1: (["1  Strided arrays", "views, broadcasting"], 0, 0),
    2: (["2  Matrix multiply", "and operation count"], 0, 1),
    3: (["3  Stable reductions", "compensated sum, lse"], 1, 0),
    4: (["4  Generator", "SplitMix64 and samplers"], 2, 0),
    5: (["5  Derivatives", "and gradients"], 0, 2),
    6: (["6  Statistics", "mean, se, bootstrap"], 2, 1),
    7: (["7  Integration", "line fit with interval"], 1, 3),
}


def render() -> str:
    d = Diagram(W, 2 * TOP + 3 * PITCH)
    m = {n: d.box(COLS[col], TOP + layer * PITCH, BOX_W, BOX_H, lines, "result" if n == 7 else "model")
         for n, (lines, col, layer) in MILESTONES.items()}

    def at(box, edge, dx):
        """A point on the top or bottom edge of a box, dx from its middle."""
        return (box[edge][0] + dx, box[edge][1])

    d.arrow(m[1]["b"], m[2]["t"])
    d.arrow(m[2]["b"], m[5]["t"], "test function", side="left")
    d.path([m[1]["l"], (MARGIN_X, m[1]["l"][1]), (MARGIN_X, m[5]["l"][1]), m[5]["l"]])
    d.arrow(at(m[3], "b", -40), at(m[5], "t", 65))
    d.arrow(at(m[3], "b", 40), at(m[6], "t", -40))
    d.arrow(at(m[4], "b", 30), at(m[6], "t", 30))
    d.arrow(m[5]["r"], at(m[7], "t", -50))
    d.arrow(at(m[6], "b", -40), at(m[7], "t", 50))
    d.path([m[4]["r"], (W - MARGIN_X, m[4]["r"][1]), (W - MARGIN_X, m[7]["r"][1]), m[7]["r"]])
    return d.svg()
