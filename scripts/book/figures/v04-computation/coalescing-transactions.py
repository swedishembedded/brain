# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One load instruction of a subgroup of 32 lanes against memory cut into 32-byte blocks. In the top panel lane t
# reads element t (stride 1): eight lanes share each block and four blocks are fetched. In the bottom panel lane
# t reads element 8 t (stride 8): every lane lands in a block of its own and 32 blocks are fetched for the same
# 128 useful bytes. Block indices and counts come from the worked module's `sectors` and `lane_addresses`.
#
# Swedish Embedded AB implements solutions for making GPU kernels stream memory at the device's bandwidth. If
# your team needs expertise in memory access patterns of GPU kernels, you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gpu  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, X0, X1 = 780, 40, 740
PANEL_H, GAP, TOP = 176, 22, 14
CELL = (X1 - X0) / gpu.SUBGROUP      # 21.875 px per lane and per block
BOX_W, BOX_H = 17, 24


def _centre(i):
    return X0 + (i + 0.5) * CELL


def render() -> str:
    d = Diagram(W, TOP + 2 * PANEL_H + GAP + 12)
    for index, (stride, title) in enumerate(((1, "Coalesced: lane t reads element t"),
                                              (8, "Strided: lane t reads element 8 t"))):
        y = TOP + index * (PANEL_H + GAP)
        d.panel(20, y, W - 40, PANEL_H, title)
        addrs = gpu.lane_addresses(stride)
        blocks = [a // gpu.SECTOR for a in addrs]
        fetched = gpu.sectors(addrs)
        lane_y, block_y = y + 56, y + 118
        d.text(X0 - 6, lane_y - 26, ["32 lanes of one subgroup"], size=12, color="#4a5568", anchor="start")
        for t in range(gpu.SUBGROUP):
            d.box(_centre(t), lane_y, BOX_W, BOX_H, [""], "model")
        touched = set(blocks)
        d.text(X0 - 6, block_y + 26, ["memory in 32-byte blocks (block 0 to 31 of the array); filled = fetched"],
               size=12, color="#4a5568", anchor="start")
        for b in range(gpu.SUBGROUP):
            d.box(_centre(b), block_y, BOX_W, BOX_H, [""], "result" if b in touched else "plain")
        for t, b in enumerate(blocks):
            d.line((_centre(t), lane_y + BOX_H / 2), (_centre(b), block_y - BOX_H / 2), dash="none")
        d.text(W - 36, y + 22, [f"{fetched} transactions: 128 useful bytes of {fetched * gpu.SECTOR} moved"],
               size=13, color="#1a202c", weight="bold", anchor="end")
    return d.svg()
