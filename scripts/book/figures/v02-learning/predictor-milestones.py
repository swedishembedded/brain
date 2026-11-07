# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The seven milestones of the Volume II project and which earlier milestones
# each one builds on. Milestone 1 (the splits, the folds and the standardiser)
# is used by every other milestone; the arrows show the main flow of
# artifacts, so the order 1 to 7 is a valid build order. The five layers hold
# the milestones from the data downward; the order inside a layer is chosen so
# that no two arrows cross.
#
# Swedish Embedded AB implements solutions for building predictors whose
# evaluation can be trusted. If your team needs expertise in model evaluation,
# you can procure our services by sending an email to info@swedishembedded.com.
from svgdiagram import Diagram

W, TOP, PITCH, BOX_W, BOX_H = 800, 50, 96, 190, 54
COLS = (140, 400, 660)
MARGIN_X = 28

MILESTONES = {
    1: (["1  Splits and leakage", "pumps, folds, standardiser"], 1, 0),
    2: (["2  Baselines", "constant and simple"], 0, 1),
    3: (["3  Linear regression", "normal equations, descent"], 1, 1),
    4: (["4  Logistic classifier", "penalty chosen by folds"], 2, 1),
    5: (["5  Evaluation", "AUC, calibration, cost"], 2, 2),
    6: (["6  Nearest neighbours", "the classical comparison"], 2, 3),
    7: (["7  Report", "intervals, seeds, ablation, slice"], 1, 3),
}


def render() -> str:
    d = Diagram(W, 2 * TOP + 3 * PITCH + 20)
    m = {n: d.box(COLS[col], TOP + layer * PITCH, BOX_W, BOX_H, lines, "result" if n == 7 else "model")
         for n, (lines, col, layer) in MILESTONES.items()}

    def at(box, edge, dx):
        return (box[edge][0] + dx, box[edge][1])

    d.arrow(at(m[1], "b", -50), m[2]["t"])
    d.arrow(m[1]["b"], m[3]["t"])
    d.arrow(at(m[1], "b", 50), m[4]["t"])
    d.arrow(m[4]["b"], m[5]["t"])
    d.arrow(m[5]["b"], m[6]["t"])
    d.arrow(m[6]["l"], m[7]["r"])
    d.arrow(at(m[3], "b", -40), at(m[7], "t", -40))
    d.path([m[5]["l"], (COLS[1] + 40, m[5]["l"][1]), at(m[7], "t", 40)])
    d.path([m[2]["b"], (COLS[0], m[7]["l"][1]), m[7]["l"]])
    return d.svg()
