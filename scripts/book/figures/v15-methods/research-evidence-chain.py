# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The path from a registered protocol to a typeset number in brain's spectral-lm paper, in the chapter "From Hypothesis
# to Paper": the driver runs a registered plan only on a committed tree, each run leaves a record naming its commit,
# command, seed and device, the records are committed and feed both the next plan and the table generator, and the
# paper's tables, quoted numbers, verdicts and draft status are generated from them.
#
# Swedish Embedded AB implements solutions for research results that others can check and rebuild for its clients. If
# your team needs expertise in reproducible measurement pipelines and papers built from records, you can procure our
# services by sending an email to info@swedishembedded.com.
from svgdiagram import Diagram

W = 820
COLS = (110, 310, 510, 710)
BOX_W, BOX_H = 176, 84
ROW1, ROW2 = 130, 290


def render() -> str:
    d = Diagram(W, 360)
    top = [
        ["registered protocol", "hypotheses, margin,", "arms, budget, seeds"],
        ["driver runs a plan", "refuses a tree with", "uncommitted changes"],
        ["one record per run", "commit, command, seed,", "device, measurements"],
        ["records committed", "under data/<plan>/", "(null if not measured)"],
    ]
    bottom = [
        ["table generator", "reads every record,", "types no number"],
        ["tables and macros", "quoted numbers, verdicts,", "draft or final status"],
        ["paper sections", "cite the macros,", "never a typed value"],
        ["PDF", "rebuilt by one", "make target"],
    ]
    kinds = ("plain", "model", "result", "result")
    a = [d.box(x, ROW1, BOX_W, BOX_H, lines, k) for x, lines, k in zip(COLS, top, kinds)]
    b = [d.box(x, ROW2, BOX_W, BOX_H, lines, k) for x, lines, k in zip(COLS, bottom, ("model", "result", "plain", "plain"))]
    for i in range(3):
        d.arrow(a[i]["r"], a[i + 1]["l"])
        d.arrow(b[i]["r"], b[i + 1]["l"])
    mid = (ROW1 + ROW2) / 2
    d.path([a[3]["b"], (COLS[3], mid), (COLS[0], mid), b[0]["t"]])
    # A later plan reads committed records: the learning rate of the comparison runs is chosen from the selection runs.
    up = ROW1 - BOX_H / 2 - 34
    d.path([a[3]["t"], (COLS[3], up), (COLS[1], up), a[1]["t"]], dash="5 3")
    d.text((COLS[1] + COLS[3]) / 2, up - 12, ["the next plan reads committed records (the chosen learning rate)"], size=12)
    return d.svg()
