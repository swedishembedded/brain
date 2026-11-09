# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The seven milestones of the Volume V project in build order, as a snake: the tokenizer, the data pipeline and the model across
# the top, training down the right edge, then generation, the cache and the work report right to left along the bottom. Every arrow
# points from a milestone to one that uses its result, and each box names the check that ends the milestone.
#
# Swedish Embedded AB implements solutions for building language models whose every stage is checked. If your team needs expertise
# in language-model pipelines and cache correctness, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram  # noqa: E402

W, H = 840, 340
COLS = (130, 420, 710)
ROWS = (70, 170, 270)
BOX_W, BOX_H = 220, 66

MILESTONES = {
    1: (["1  Tokenizer", "merge-list digest"], 0, 0, "model"),
    2: (["2  Data pipeline", "first batch"], 1, 0, "model"),
    3: (["3  The GPT", "logit digest, gradients"], 2, 0, "model"),
    4: (["4  Training", "stable loss, exact resume"], 2, 1, "model"),
    5: (["5  Generation", "ids for seeds"], 1, 1, "model"),
    6: (["6  KV cache", "equivalence, mutations"], 0, 1, "result"),
    7: (["7  Work report", "FLOPs, bytes, intensity"], 0, 2, "model"),
}


def render() -> str:
    d = Diagram(W, H)
    m = {n: d.box(COLS[col], ROWS[row], BOX_W, BOX_H, lines, kind) for n, (lines, col, row, kind) in MILESTONES.items()}
    d.arrow(m[1]["r"], m[2]["l"])
    d.arrow(m[2]["r"], m[3]["l"])
    d.arrow(m[3]["b"], m[4]["t"])
    d.arrow(m[4]["l"], m[5]["r"])
    d.arrow(m[5]["l"], m[6]["r"])
    d.arrow(m[6]["b"], m[7]["t"])
    return d.svg()
