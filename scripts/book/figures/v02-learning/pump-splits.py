# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# How the 360 pumps of the project are divided. The unit of the split is the
# pump, never the row: site A's 300 pumps become 180 training, 60 validation
# and 60 test pumps, the 60 pumps of site B form the shift slice that nothing
# looks at before the report, and the 180 training pumps are dealt into five
# folds of 36. Box widths are proportional to the number of pumps.
#
# Swedish Embedded AB implements solutions for evaluation protocols that keep
# related records on one side of a split. If your team needs expertise in
# honest model evaluation, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
from svgdiagram import Diagram  # noqa: E402
import project_predictor as pp  # noqa: E402

W, X0, PX = 800, 40, 2.0          # left edge of the bars and pixels per pump
BAR_H = 44


def render() -> str:
    a, b = pp.G_SITE_A, pp.G_SITE_B
    tr, va, te = pp.N_TRAIN_PUMPS, pp.N_VAL_PUMPS, pp.N_TEST_PUMPS
    k = pp.K_FOLDS
    d = Diagram(W, 300)

    def bar(row_y, start, count, lines, kind):
        return d.box(X0 + (start + count / 2) * PX, row_y, count * PX - 6, BAR_H, lines, kind)

    d.text(X0, 20, [f"{a + b} pumps, {pp.ROWS_PER_PUMP} rows each"], size=13, weight="bold", anchor="start")
    # a bracket over the three sets that come from site A
    y = 48
    d.line((X0, y), (X0 + a * PX - 6, y), dash=None)
    d.line((X0, y), (X0, y + 8), dash=None)
    d.line((X0 + a * PX - 6, y), (X0 + a * PX - 6, y + 8), dash=None)
    d.text(X0 + a * PX / 2, y - 10, [f"site A: {a} pumps"], size=12)
    d.line((X0 + a * PX + 6, y), (X0 + (a + b) * PX, y), dash=None)
    d.line((X0 + a * PX + 6, y), (X0 + a * PX + 6, y + 8), dash=None)
    d.line((X0 + (a + b) * PX, y), (X0 + (a + b) * PX, y + 8), dash=None)
    d.text(X0 + (a + b / 2) * PX, y - 10, [f"site B: {b} pumps"], size=12)
    cy = 92
    train = bar(cy, 0, tr, [f"training: {tr} pumps"], "model")
    bar(cy, tr, va, [f"validation: {va}"], "model")
    bar(cy, tr + va, te, [f"test: {te}"], "result")
    bar(cy, a, b, [f"shift: {b}"], "accent")
    cap = cy + BAR_H / 2 + 16
    d.text(X0 + 12, cap, ["fit; folds choose lambda, k"], size=12, anchor="start")
    d.text(X0 + (tr + va / 2) * PX, cap, ["scale a"], size=12)
    d.text(X0 + (tr + va + te / 2) * PX, cap, ["read once"], size=12)
    d.text(X0 + (a + b / 2) * PX, cap, ["read last"], size=12)
    # the five folds sit under the training box and span exactly its width
    fy = 192
    for j in range(k):
        bar(fy, j * tr / k, tr / k, [f"fold {j}", f"{tr // k}"], "plain")
    d.line((X0, cy + BAR_H / 2 + 2), (X0, fy - BAR_H / 2 - 2), dash="4 3")
    d.line((X0 + tr * PX - 6, cy + BAR_H / 2 + 2), (X0 + tr * PX - 6, fy - BAR_H / 2 - 2), dash="4 3")
    d.text(X0 + tr * PX + 14, fy - 6, ["each fold: whole pumps of the training set,", "scored by models fitted on the other four"],
           size=12, anchor="start")
    d.text(X0, 262, ["Each box is a set of whole pumps: the four rows of a pump are always on the same side."],
           size=12, anchor="start")
    return d.svg()
