# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The loop of tiled attention for one block of query rows: the query block is read once, the key and value tiles stream through
# shared memory one after another, the B_q by B_k tile of scores exists only on chip and updates the running state (m, l, o), and the
# output block and the log-sum-exp are written once at the end. The sizes in the labels come from `flash_attention.SIM_*`, the
# case that the chapter's simulated kernel runs.
#
# Swedish Embedded AB implements solutions for attention kernels that keep the score matrix out of global memory. If your team needs
# expertise in fused attention for long sequences, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flash_attention as fa  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 400


def render() -> str:
    bq, bk = fa.SIM_BQ, fa.SIM_BK
    d = Diagram(W, H)
    d.text(W / 2, 24, [f"One workgroup of tiled attention: {bq} query rows against key and value tiles of {bk} rows"], size=14, weight="bold")
    d.panel(12, 46, 228, 330, "Global memory")
    d.panel(262, 46, 492, 330, "On chip: registers and shared memory")
    d.panel(776, 46, 212, 330, "Global memory")
    q = d.box(126, 130, 180, 54, [f"Q block i", f"{bq} rows, read once"], "plain")
    kv = d.box(126, 250, 180, 54, ["K tile j, V tile j", f"{bk} rows each, read per tile"], "plain")
    out = d.box(882, 250, 180, 64, ["O block i = o / l", "log-sum-exp = m + ln l", "written once"], "result")
    qb = d.box(360, 130, 150, 54, ["q rows", "registers"], "model")
    kvs = d.box(360, 250, 150, 54, ["K_j, V_j", "shared memory"], "model")
    sc = d.box(540, 250, 130, 54, [f"S_ij, {bq} x {bk}", "never stored"], "accent")
    st = d.box(690, 250, 100, 84, ["m, l, o", "registers", "per row"], "result")
    d.arrow(q["r"], qb["l"], "once")
    d.arrow(kv["r"], kvs["l"], f"j = 0 .. {fa._ceil_div(fa.SIM_T, bk) - 1}")
    d.path([qb["r"], (sc["t"][0], qb["r"][1]), sc["t"]])
    d.arrow(kvs["r"], sc["l"], "q K^T")
    d.arrow(sc["r"], st["l"])
    d.arrow(st["r"], out["l"], "once")
    d.path([(690, 292), (690, 330), (360, 330), (360, 277)], color=MUTED, dash="5 3")
    d.text(525, 346, ["the next tile: rescale o and l by exp(m_old - m_new), add the tile"], size=12, color=MUTED)
    return d.svg()
