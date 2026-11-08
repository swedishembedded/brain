# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A recurrent cell folded and unrolled over three tokens, the sequence `the cats near` of
# `recurrent_networks.unroll`. Above, the one cell with its state fed back; below, the same cell copied once per
# token, the state passed along and the gradient of the loss passed back (red) through the same matrix W each time.
#
# Swedish Embedded AB implements solutions for sequence models that read and write streams of data. If your team
# needs expertise in recurrent networks and their training, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import INK, RED, Diagram, ellipse_x  # noqa: E402

W, H = 960, 520
TOKENS = ["the", "cats", "near"]


def render() -> str:
    d = Diagram(W, H)

    # folded
    d.panel(10, 10, W - 20, 170, "Folded: one cell, used again for every token")
    cell = d.box(440, 108, 250, 52, ["h(t) = tanh(W h(t-1) + u(x(t)) + b)"], "model")
    tok = d.box(160, 108, 120, 36, ["token x(t)"], "accent")
    out = d.oval(780, 108, 100, 40, ["state h(t)"], "result")
    d.arrow(tok["r"], cell["l"], "u(x(t))", side="above")
    d.arrow(cell["r"], out["l"])
    d.path([(out["t"][0], out["t"][1]), (out["t"][0], 62), (cell["t"][0], 62), (cell["t"][0], cell["t"][1])], radius=8)
    d.text(610, 52, ["h(t) becomes h(t-1) at the next token"], size=12, color="#4a5568")

    # unrolled
    d.panel(10, 194, W - 20, 316, "Unrolled over three tokens: the same W, u and b in every box")
    y = 300
    xs_h = [70, 250, 430, 610]          # h(0) .. h(3)
    xs_s = [160, 340, 520]              # step boxes
    h_nodes = [d.oval(x, y, 54, 40, [f"h({i})"], "result") for i, x in enumerate(xs_h)]
    steps = [d.box(x, y, 80, 48, [f"step {i + 1}", "tanh"], "model") for i, x in enumerate(xs_s)]
    loss = d.box(810, y, 96, 48, ["loss L", "(h(3), y)"], "plain")
    dy = 11
    for i in range(3):
        a = (ellipse_x(h_nodes[i]["ell"], "r", dy), y + dy)
        d.arrow(a, (steps[i]["l"][0], y + dy))
        d.arrow((steps[i]["r"][0], y + dy), (ellipse_x(h_nodes[i + 1]["ell"], "l", dy), y + dy))
    d.arrow((ellipse_x(h_nodes[3]["ell"], "r", dy), y + dy), (loss["l"][0], y + dy))
    # gradient flow, leftwards, above the forward arrows
    g = -11
    d.arrow((loss["l"][0], y + g), (ellipse_x(h_nodes[3]["ell"], "r", -g), y + g), color=RED)
    for i in (2, 1, 0):
        d.arrow((ellipse_x(h_nodes[i + 1]["ell"], "l", -g), y + g), (steps[i]["r"][0], y + g), color=RED)
        d.arrow((steps[i]["l"][0], y + g), (ellipse_x(h_nodes[i]["ell"], "r", -g), y + g), color=RED)
    d.text(722, y - 32, ["dh(3) = h(3) - y"], size=12, color=RED)
    d.text(340, y - 50, ["dh(t-1) = W^T ((1 - h(t)^2) * dh(t)) at every step"], size=12, color=RED)
    # tokens under the steps
    for i, x in enumerate(xs_s):
        t = d.box(x, y + 100, 84, 34, [f"`{TOKENS[i]}`".replace("`", "")], "accent")
        d.arrow(t["t"], steps[i]["b"], "u(x(%d))" % (i + 1), side="right")
    # the shared parameters
    d.text(W / 2, y + 168, ["each box contributes d a(t) h(t-1)^T to dW, d a(t) to db and d a(t) to the column u(x(t)); the three terms are added"],
           size=12, color="#4a5568")
    d.text(W / 2, y + 190, ["forward: black arrows, h(t) from h(t-1) and the token; backward: red arrows, d h(t-1) from d h(t)"], size=12, color="#4a5568")
    return d.svg()
