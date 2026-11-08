# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Index mapping of a transpose kernel with one invocation per output element, for a [3, 8] matrix stored row-major and
# transposed to [8, 3]. The top row is the output in the order it is written (invocation g writes y[g]); the bottom row
# is the input storage; a line joins output g to the input offset it reads. Writes are consecutive, reads jump by 4
# (the row length). For a toy subgroup of 4 invocations and memory blocks of 4 numbers the first subgroup is outlined and
# the input blocks it touches are filled. The offsets come from the worked module's `strided_toy`.
#
# Swedish Embedded AB implements solutions for GPU kernels whose memory accesses match the layout of their tensors. If your
# team needs expertise in strided tensor access on GPUs, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import compute_kernels as ck  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 780, 340
X0, X1 = 60, 720
LANES, BLOCK = 4, 4           # toy subgroup size and memory block, in numbers


def render() -> str:
    m, n, reads = ck.strided_toy()
    total = m * n
    cell = (X1 - X0) / total
    cx = lambda i: X0 + (i + 0.5) * cell
    out_y, in_y, bh = 104, 244, 34
    d = Diagram(W, H)
    d.panel(20, 14, W - 40, H - 28, f"Transposing a [{m}, {n}] matrix: invocation g writes y[g] and reads x[reads[g]]")
    d.text(X0 - 2, out_y - 34, [f"output y, written in order g = 0 .. {total - 1}"], size=12, color="#4a5568", anchor="start")
    d.text(X0 - 2, in_y + 36, [f"input x in row-major storage, offsets 0 .. {total - 1}; memory blocks of {BLOCK} numbers (dashed lines)"],
           size=12, color="#4a5568", anchor="start")
    first = set(reads[:LANES])
    blocks = {r // BLOCK for r in reads[:LANES]}
    for g in range(total):
        d.box(cx(g), out_y, cell - 4, bh, [str(g)], "accent" if g < LANES else "plain")
    for off in range(total):
        kind = "result" if off // BLOCK in blocks else "plain"
        d.box(cx(off), in_y, cell - 4, bh, [str(off)], kind)
    for b in range(1, total // BLOCK):
        x = X0 + b * BLOCK * cell
        d.line((x, in_y - bh / 2 - 4), (x, in_y + bh / 2 + 4), dash="3 3")
    for g, r in enumerate(reads):
        d.arrow((cx(g), out_y + bh / 2), (cx(r), in_y - bh / 2), None, gap=2,
                color="#1a202c" if g < LANES else "#a0aec0")
    d.text(X0 - 2, in_y + 58, [f"The first {LANES} invocations (filled input blocks) read offsets {', '.join(str(r) for r in reads[:LANES])}: "
                               f"{len(blocks)} blocks for {LANES} numbers; reading rows instead would touch 1 block"],
           size=12, color="#1a202c", anchor="start")
    return d.svg()
