# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A plain layer and a residual layer as block diagrams: the same linear map and activation, and
# in the residual layer a scale s and a sum with the layer's own input. The sum is what gives the
# gradient a path that does not pass through the weights.
#
# Swedish Embedded AB implements solutions for finding out why a training run misbehaves. If your
# team needs expertise in training diagnostics for neural networks, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram  # noqa: E402

W, H = 900, 380
OVAL_W, BOX_W, GAP = 62, 100, 30


def row(d_, cy, specs, x0):
    """Place nodes left to right from x0 on the row cy; returns the nodes."""
    nodes, cx = [], x0
    for kind, lines, style, w in specs:
        node = (d_.oval if kind == "oval" else d_.box)(cx + w / 2, cy, w, 56, lines, style)
        nodes.append(node)
        cx += w + GAP
    for a, b in zip(nodes, nodes[1:]):
        d_.arrow(a["r"], b["l"])
    return nodes


def render() -> str:
    d_ = Diagram(W, H)
    d_.panel(10, 10, 880, 130, "plain layer:  a' = g(W a + b),   gradient of a:  W^T D da'")
    row(d_, 85, [("oval", ["a"], "result", OVAL_W), ("box", ["Linear", "W, b"], "model", BOX_W), ("oval", ["z"], "plain", OVAL_W),
                 ("box", ["g"], "accent", BOX_W), ("oval", ["a'"], "result", OVAL_W)], 60)
    d_.panel(10, 160, 880, 210, "residual layer:  a' = a + s g(W a + b),   gradient of a:  da' + s W^T D da'")
    cy = 305
    nodes = row(d_, cy, [("oval", ["a"], "result", OVAL_W), ("box", ["Linear", "W, b"], "model", BOX_W), ("oval", ["z"], "plain", OVAL_W),
                         ("box", ["g"], "accent", BOX_W), ("box", ["times s"], "model", BOX_W), ("oval", ["+"], "plain", 40),
                         ("oval", ["a'"], "result", OVAL_W)], 60)
    a, plus = nodes[0], nodes[5]
    top = 222
    d_.path([(a["t"][0], a["t"][1]), (a["t"][0], top), (plus["t"][0], top), (plus["t"][0], plus["t"][1])])
    d_.text((a["t"][0] + plus["t"][0]) / 2, top - 14, ["identity path: no weights, no slope"], size=12, color="#4a5568")
    return d_.svg()
