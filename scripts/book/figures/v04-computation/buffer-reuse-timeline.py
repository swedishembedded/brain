# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Static memory planning as a picture: time is the op index, height is the byte offset in the activation region. Each
# rectangle is one tensor from the op that writes it (or the start) to the last op that reads it, at the offset the
# planner of the worked module (`memory_plan_data`, first fit, largest first) gave it. Rectangles that share a column
# never overlap; rectangles that share bytes never share a column.
#
# Swedish Embedded AB implements solutions for fitting neural networks into the small fixed memories of edge
# accelerators. If your team needs expertise in static memory planning for NPUs, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import npu  # noqa: E402
from svgdiagram import Diagram, FILL, INK, MUTED  # noqa: E402
from svgplot import fmt  # noqa: E402

W, H = 880, 392
X0, CW = 112, 100            # left edge of the first op column, column width
Y_TOP, Y_BASE = 60, 316     # y of offset peak and of offset 0
KINDS = ("model", "accent", "result", "plain")


def render() -> str:
    live, sizes, offset, at_op = npu.memory_plan_data()
    peak = max(at_op)
    top = max(offset[t] + sizes[t] for t in sizes)
    total = sum(sizes.values())
    scale = (Y_BASE - Y_TOP) / top
    n_cols = len(npu.PLAN_OPS) + 1
    d = Diagram(W, H)
    d.panel(20, 12, W - 40, H - 24, "Tensor lifetimes and the bytes they share")
    # grid: op columns and the byte axis
    for i in range(n_cols):
        label = f"op {i}" if i < n_cols - 1 else "after"
        d.text(X0 + CW * (i + 0.5), Y_BASE + 20, [label], size=12, color=MUTED)
        if i:
            d.line((X0 + CW * i, Y_TOP - 6), (X0 + CW * i, Y_BASE), dash="2 4")
    d.line((X0, Y_BASE), (X0 + CW * n_cols, Y_BASE), dash="1 0")
    for b in (0, 4096, 8192, 12288):
        y = Y_BASE - b * scale
        d.text(X0 - 10, y, [f"{b:,}"], size=11, color=MUTED, anchor="end")
        d.line((X0 - 4, y), (X0, y), dash="1 0")
    # the tensors
    for k, t in enumerate(sorted(sizes)):
        a, b = live[t]
        x, w = X0 + CW * a + 3, CW * (b - a + 1) - 6
        y1 = Y_BASE - (offset[t] + sizes[t]) * scale
        h = sizes[t] * scale
        d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y1)}" width="{fmt(w)}" height="{fmt(h)}" rx="3" '
                       f'fill="{FILL[KINDS[k % 4]]}" stroke="{MUTED}" stroke-width="1.2"/>')
        d.text(x + w / 2, y1 + h / 2, [f"{t}  {sizes[t]:,} B"], size=12)
    # the peak line and the cost without reuse
    ypk = Y_BASE - peak * scale
    d.line((X0, ypk - 0.5), (X0 + CW * n_cols, ypk - 0.5), dash="6 3")
    d.text(X0 + CW * n_cols + 8, ypk - 6, [f"{peak:,} B alive", "at op 0"], size=12, color=INK, anchor="start")
    d.text(X0 + CW * n_cols + 8, Y_BASE - 44, [f"plan ends at", f"{top:,} B,", "without reuse", f"{total:,} B"], size=12,
           color=MUTED, anchor="start")
    d.text(54, (Y_TOP + Y_BASE) / 2, ["byte", "offset"], size=12, color=MUTED, anchor="middle")
    return d.svg()
