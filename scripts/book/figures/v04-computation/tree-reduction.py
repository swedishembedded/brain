# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two trees a workgroup of 16 invocations builds when it adds the sixteen numbers of the chapter's first example.
# Top panel: neighbours first (kernels A and B: invocation l adds the value of l + s for s = 1, 2, 4, 8). Bottom panel:
# sequential addressing (kernel C: invocation l < s adds the value of l + s, s = 8, 4, 2, 1). Both are balanced trees
# of four levels with fifteen additions; they pair the numbers differently, which is why float32 sums differ. The nodes are
# the values the kernels hold, built by the worked module from the kernels' own addressing rules.
#
# Swedish Embedded AB implements solutions for fast parallel reductions on GPUs. If your team needs expertise in
# reduction kernels for machine learning, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import reductions as rd  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

PITCH, BOX_W, BOX_H, ROW = 40, 32, 26, 54
X0 = 190                       # left edge of the leaf grid
PANEL_H = 4 * ROW + 100
W = X0 + 16 * PITCH + 34
H = 2 * PANEL_H + 30


def _leaves(node):
    return [node] if node[1] is None else _leaves(node[1]) + _leaves(node[2])


def _height(node):
    return 0 if node[1] is None else 1 + _height(node[1])


def _draw(d, top, variant, title, strides):
    d.panel(14, top, W - 28, PANEL_H, title)
    n = len(rd.X16)
    root = rd.tree_structure(rd.X16, variant)
    x_of = {leaf[3]: X0 + (k + 0.5) * PITCH for k, leaf in enumerate(_leaves(root))}
    base = top + 52 + 4 * ROW          # y of the leaf row centre

    def place(node):
        """(x, y, edge midpoints) of a node; draws the node and the arrows from its children."""
        h = _height(node)
        y = base - h * ROW
        if h == 0:
            x = x_of[node[3]]
            kind = "plain"
            pts = d.box(x, y, BOX_W, BOX_H, [str(node[0])], kind)
            d.text(x, y + BOX_H / 2 + 11, [str(node[3])], size=10, color="#4a5568")
            return x, pts
        xl, pl = place(node[1])
        xr, pr = place(node[2])
        x = (xl + xr) / 2
        pts = d.box(x, y, BOX_W, BOX_H, [str(node[0])], "result" if h == 4 else "model")
        ymid = y + BOX_H / 2 + (ROW - BOX_H) / 2
        for xc, pc in ((xl, pl), (xr, pr)):
            d.path([pc["t"], (xc, ymid), (x, ymid), pts["b"]], radius=6)
        return x, pts

    place(root)
    for k, s in enumerate(strides):
        y = base - (k + 1) * ROW
        adds = n // (2 * s) if variant == "interleaved" else s
        d.text(34, y - 4, [f"stride s = {s}", f"{adds} addition" + ("" if adds == 1 else "s")], size=12, color="#4a5568",
               anchor="start")
    d.text(34, base - 4, ["load", f"{n} numbers x_l"], size=12, color="#4a5568", anchor="start")


def render() -> str:
    d = Diagram(W, H)
    _draw(d, 14, "interleaved", "Kernels A and B: neighbours first, invocation l adds the value of l + s",
          rd.STRIDES_UP(len(rd.X16)))
    _draw(d, 14 + PANEL_H + 2, "sequential", "Kernel C: sequential addressing, invocation l < s adds the value of l + s",
          rd.STRIDES_DOWN(len(rd.X16)))
    return d.svg()
