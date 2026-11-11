# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One workgroup of a tiled matrix product C = A B: for n = 16 and T = 4 the matrices are 4 x 4 grids of T x T
# blocks. The workgroup that owns block (3, 1) of C walks along block row 3 of A and block column 1 of B; the
# picture shows its last step, s = 3, where it copies the two blocks into shared memory and every invocation
# does T multiply-adds from there. The counts under the picture come from the worked module's `tiled_loads`,
# so the figure and the chapter's table cannot disagree.
#
# Swedish Embedded AB implements solutions for blocking dense linear algebra around the memory hierarchy of a
# GPU. If your team needs expertise in matrix multiplication kernels for machine learning, you can procure our
# services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matmul_fast as mf  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

N, T = 16, 4
BLOCKS = N // T            # blocks per side
CELL = 34                  # px per block
W, H = 780, 398
ROW_BAND, COL_BAND, STEP = 3, 1, 3      # the workgroup's block row of A, block column of B, and the step drawn


def render() -> str:
    d = Diagram(W, H)
    d.panel(20, 14, W - 40, H - 28, f"One workgroup of a tiled product, n = {N}, T = {T}: it owns block ({ROW_BAND}, {COL_BAND}) of C")
    side = BLOCKS * CELL
    y0 = 84
    xs = {"A": 102, "B": 322, "C": 542}
    kind = {}
    for j in range(BLOCKS):
        kind[("A", ROW_BAND, j)] = "accent"
        kind[("B", j, COL_BAND)] = "accent"
    kind[("A", ROW_BAND, STEP)] = "result"
    kind[("B", STEP, COL_BAND)] = "result"
    kind[("C", ROW_BAND, COL_BAND)] = "result"
    for name, caption in (("A", f"A [{N}, {N}]"), ("B", f"B [{N}, {N}]"), ("C", f"C [{N}, {N}]")):
        x0 = xs[name]
        d.text(x0 + side / 2, y0 - 22, [caption], size=13, weight="bold")
        for i in range(BLOCKS):
            for j in range(BLOCKS):
                d.swatch(x0 + j * CELL + 1, y0 + i * CELL + 1, CELL - 2, CELL - 2, kind.get((name, i, j), "plain"))
        for (nm, i, j), k in kind.items():
            if nm == name and k == "result":
                d.text(x0 + j * CELL + CELL / 2, y0 + i * CELL + CELL / 2, [f"{i},{j}"], size=11)
    mid = y0 + side / 2
    d.text((xs["A"] + side + xs["B"]) / 2, mid, ["x"], size=20, color="#4a5568")
    d.text((xs["B"] + side + xs["C"]) / 2, mid, ["="], size=20, color="#4a5568")
    # shared memory box under the grids, fed from the two active blocks, feeding the C block
    by = 300
    box = d.box(380, by, 372, 56, [f"shared memory of the workgroup: a_tile and b_tile, {T} x {T} each", "(a barrier before and after the multiply-adds)"],
                "model")
    ax = xs["A"] + STEP * CELL + CELL / 2
    bx = xs["B"] + COL_BAND * CELL + CELL / 2
    cx = xs["C"] + COL_BAND * CELL + CELL / 2
    bottom = y0 + side
    d.arrow((ax, bottom), (ax, box["t"][1]), label=f"{T * T} numbers of A", side="left")
    d.arrow((bx, bottom), (bx, box["t"][1]), label=f"{T * T} numbers of B", side="right")
    d.path([box["r"], (cx, box["r"][1]), (cx, bottom)])
    d.text(cx + 12, (box["r"][1] + bottom) / 2, [f"{T} multiply-adds", "per output"], size=12, color="#4a5568", anchor="start")
    # counts
    one_wg = mf.tiled_loads(T, T, N, T)
    naive = T * T * 2 * N
    d.text(W / 2, 360, [f"This workgroup loads {one_wg} numbers over {BLOCKS} steps ({2 * T * T} per step) to compute its {T * T} outputs.",
                        f"One invocation per output would load {naive}: every number loaded is used {naive // one_wg} times."],
           size=12, color="#4a5568")
    return d.svg()
