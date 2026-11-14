# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The capstone's eight requests on a server of three slots, iteration by iteration, under continuous and under static batching.
# Every cell is computed by the worked module's server (`capstone.serve`): which request holds the slot, whether it was prefilled
# (P) or decoded one token (D) in that iteration, or held its slot without running ((j), static batching only).
#
# Swedish Embedded AB implements solutions for scheduling many requests onto one accelerator. If your team needs expertise in
# continuous batching and serving cost then you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import capstone  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

CELL_W, CELL_H = 42, 30
LEFT, TOP = 150, 64
PANEL_GAP = 70


def render() -> str:
    grids = capstone.schedule_grids()
    width = max(len(r["timeline"]) for _, r in grids)
    W = LEFT + width * CELL_W + 30
    H = TOP + len(grids) * (capstone.SLOTS * CELL_H + PANEL_GAP) - 20
    d = Diagram(W, H)
    for g, (policy, r) in enumerate(grids):
        y0 = TOP + g * (capstone.SLOTS * CELL_H + PANEL_GAP)
        d.text(LEFT, y0 - 34, [f"{policy} batching: {r['iterations']} iterations run"], size=13, weight="bold", anchor="start")
        for k in range(width):
            d.text(LEFT + k * CELL_W + CELL_W / 2, y0 - 12, [str(k)], size=11, color=MUTED)
        for s in range(capstone.SLOTS):
            cy = y0 + s * CELL_H + CELL_H / 2
            d.text(LEFT - 14, cy, [f"slot {s}"], size=12, anchor="end")
            for k, row in enumerate(r["timeline"]):
                cell = row[s]
                if cell is None:
                    continue
                j, kind = cell
                label = {"p": f"P{j}", "d": f"D{j}", "held": f"({j})"}[kind]
                fill = {"p": "result", "d": "accent", "held": "plain"}[kind]
                d.box(LEFT + k * CELL_W + CELL_W / 2, cy, CELL_W - 4, CELL_H - 4, [label], fill)
    d.text(LEFT, H - 22, ["P j: request j prefilled, D j: one token of request j decoded, (j): slot held by request j, nothing run; columns are iterations"],
           size=12, color=MUTED, anchor="start")
    return d.svg()
