# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Three dropout masks over the same six activations at p = 0.5. The top row is eval mode: every activation passes
# unchanged. In each mask row a kept unit passes a_j / (1 - p) = 2 a_j and a dropped unit passes 0, so the layer sends a
# different random subset of its units, scaled up, on each training step; the sum of the row changes with the mask but its
# average over masks is the sum of the top row. The numbers come from worked module `regularization_nn`
# (functions `dropout_mask_data`, `dropout_masks`).
#
# Swedish Embedded AB implements solutions for models that keep working on data they have not seen. If your team needs
# expertise in regularizing neural networks, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import regularization_nn as rg  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, LEFT, COL0, COL_STEP = 800, 20, 168, 86
BOX_W, BOX_H, ROW_STEP, TOP = 70, 36, 58, 86


def fmt(v):
    return f"{v + 0.0:g}"


def render() -> str:
    a, p, masks = rg.dropout_mask_data()
    scale = 1.0 / (1.0 - p)
    rows = 1 + len(masks)
    height = TOP + rows * ROW_STEP + 56
    d = Diagram(W, height)
    d.panel(LEFT, 10, W - 2 * LEFT, height - 20, f"Dropout at p = {p:g}: the output of a layer of six units is a_j m_j / (1 - p)")
    sum_x = COL0 + 5 * COL_STEP + 118
    d.text(sum_x, TOP - 12, ["sum of the row"], size=12, color="#4a5568")
    for j in range(len(a)):
        d.text(COL0 + j * COL_STEP, TOP - 12, [f"unit {j}"], size=12, color="#4a5568")
    # eval row
    cy = TOP + 22
    d.text(LEFT + 62, cy, ["eval mode"], anchor="middle")
    for j, x in enumerate(a):
        d.box(COL0 + j * COL_STEP, cy, BOX_W, BOX_H, [fmt(x)], "model")
    d.text(sum_x, cy, [fmt(sum(a))])
    for r, (draws, m) in enumerate(masks, start=1):
        cy = TOP + 22 + r * ROW_STEP
        d.text(LEFT + 62, cy, [f"mask {r}"], anchor="middle")
        total = 0.0
        for j, (x, keep) in enumerate(zip(a, m)):
            v = x * keep * scale
            total += v
            d.box(COL0 + j * COL_STEP, cy, BOX_W, BOX_H, [fmt(v) if keep else "0"], "accent" if keep else "plain")
        d.text(sum_x, cy, [fmt(total)])
    ly = height - 34
    d.swatch(COL0 - BOX_W / 2, ly - 9, 18, 18, "accent")
    d.text(COL0 - BOX_W / 2 + 28, ly, [f"kept: a_j / (1 - p) = {scale:g} a_j"], size=12, anchor="start")
    d.swatch(COL0 + 2.6 * COL_STEP, ly - 9, 18, 18, "plain")
    d.text(COL0 + 2.6 * COL_STEP + 28, ly, ["dropped: 0"], size=12, anchor="start")
    d.swatch(COL0 + 4.2 * COL_STEP, ly - 9, 18, 18, "model")
    d.text(COL0 + 4.2 * COL_STEP + 28, ly, ["eval mode: unchanged"], size=12, anchor="start")
    return d.svg()
