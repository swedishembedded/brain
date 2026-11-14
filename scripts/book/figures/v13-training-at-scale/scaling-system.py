# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The system of the project "Scaling a Training Run": the 10 draws of a step are dealt to four ranks, each rank replays its rows
# through the frozen base and its adapters in segments of two layers, the ring all-reduce sums the four gradient vectors, Adam updates the
# 128 adapter numbers, and every 10 updates a generation is committed. The numbers are the worked module's constants.
#
# Swedish Embedded AB implements solutions for moving training runs from one device to many without changing what they learn. If your
# team needs expertise in data-parallel, memory-lean and resumable training then you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_scaling_training as pst  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402

W, H = 1000, 430
RANKS = 4


def render() -> str:
    d = Diagram(W, H)
    draws = d.box(110, 214, 170, 64, [f"draws of update t", f"{pst.BATCH} rows, row b to", f"rank b mod {RANKS}"], "model")
    ranks = []
    for j in range(RANKS):
        y = 100 + j * 76
        sizes = [len([b for b in range(pst.BATCH) if b % RANKS == jj]) for jj in range(RANKS)]
        ranks.append(d.box(390, y, 220, 56, [f"rank {j}: {sizes[j]} rows", "replay in segments of 2,", "adapter gradient"], "accent"))
        d.arrow(draws["r"], ranks[-1]["l"])
    ring = d.box(650, 214, 150, 64, ["ring all-reduce", "sum, divide by 10"], "result")
    for r in ranks:
        d.arrow(r["r"], ring["l"])
    adam = d.box(870, 130, 170, 64, ["Adam on the", f"{pst.P_AD} adapter numbers"], "model")
    gen = d.box(870, 300, 170, 64, [f"every {pst.EVERY} updates:", "values, state, meta"], "result")
    d.arrow(ring["r"], adam["l"])
    d.arrow(adam["b"], gen["t"], "t, d, m, v", side="right")
    d.path([adam["t"], (870, 22), (110, 22), draws["t"]], color=INK)
    d.text(490, 44, ["the next update draws the next 10 rows"], size=12, color=MUTED)
    base = d.oval(390, 404, 220, 40, [f"frozen base, {pst.P_BASE} numbers"], "plain")
    d.arrow(base["t"], ranks[-1]["b"])
    return d.svg()
