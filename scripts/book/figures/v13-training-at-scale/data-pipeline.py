# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The stages between raw sources and a training step in the chapter "Data Pipelines for Large Models": duplicates and validation
# overlap are removed, the groups are mixed by listing, each record is rendered to ids, a mask and a start offset, each record becomes
# a row, the rows are cut into batches that are sorted within a pool by length, and the batches follow an order that is a function of
# the seed and the epoch and that each worker shards.
#
# Swedish Embedded AB implements solutions for preparing, mixing and feeding training data at scale. If your team needs expertise in
# training data pipelines for large models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 300
Y1, Y2 = 62, 214
XS = (110, 340, 570, 800)   # four equal columns, from the left


def render() -> str:
    d = Diagram(W, H)
    src = d.oval(XS[0], Y1, 150, 56, ["sources", "documents, records"], "plain")
    dedup = d.box(XS[1], Y1, 150, 56, ["deduplicate", "exact and near copies"], "model")
    mix = d.box(XS[2], Y1, 150, 56, ["mix", "groups listed per share"], "model")
    fmt = d.box(XS[3], Y1, 150, 56, ["record format", "ids, mask, starts"], "accent")
    d.arrow(src["r"], dedup["l"])
    d.arrow(dedup["r"], mix["l"])
    d.arrow(mix["r"], fmt["l"])
    rows = d.box(XS[3], Y2, 150, 56, ["rows", "one record per row"], "model")
    batch = d.box(XS[2], Y2, 150, 56, ["batches", "sorted within a pool"], "model")
    order = d.box(XS[1], Y2, 150, 56, ["order and shard", "seed, epoch, worker"], "model")
    step = d.oval(XS[0], Y2, 150, 56, ["training step", "loss over the mask"], "result")
    d.arrow(fmt["b"], rows["t"], "[B, T] ids", side="right")
    d.arrow(rows["l"], batch["r"], "lengths", side="above")
    d.arrow(batch["l"], order["r"], "batch list", side="above")
    d.arrow(order["l"], step["r"], "batches", side="above")
    d.text((XS[0] + XS[3]) / 2, Y2 + 62, ["counted at every stage: records, tokens, supervised tokens, padding positions"], size=12, color=MUTED)
    return d.svg()
