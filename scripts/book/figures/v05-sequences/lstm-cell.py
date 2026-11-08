# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One step of the LSTM cell of `recurrent_networks.lstm_step`: the previous state and the token make the four
# pre-activations, which become the forget, input, candidate and output signals; the cell state c is kept by the
# forget gate and added to by the input gate, and the output gate decides how much of tanh(c) is shown as h.
#
# Swedish Embedded AB implements solutions for sequence models that must remember across long inputs. If your team
# needs expertise in gated recurrent networks, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import INK, Diagram  # noqa: E402

W, H = 800, 430
OY = 32   # the gates and the cell box sit this far below the old cell state, so its arrow has room


def render() -> str:
    d = Diagram(W, H)
    d.panel(10, 10, W - 20, H - 20, "One LSTM step: the cell state c is kept by a gate and added to by a gate")
    ys = [110 + OY, 175 + OY, 240 + OY, 330 + OY]
    fan = [150 + OY, 185 + OY, 225 + OY, 270 + OY]          # where the arrows leave the affine box
    names = [("forget gate f", "sigmoid"), ("input gate i", "sigmoid"), ("candidate g", "tanh"), ("output gate o", "sigmoid")]
    hin = d.oval(75, 150 + OY, 90, 40, ["h(t-1)"], "result")
    xin = d.oval(75, 260 + OY, 90, 40, ["token x(t)"], "accent")
    aff = d.box(225, 205 + OY, 110, 150, ["a = W h(t-1)", "+ u(x(t)) + b", "(4H numbers,", "in four blocks)"], "model")
    d.arrow(hin["r"], (aff["l"][0], 150 + OY))
    d.arrow(xin["r"], (aff["l"][0], 260 + OY))
    gates = [d.box(385, y, 120, 46, list(n), "accent") for y, n in zip(ys, names)]
    for yy, g in zip(fan, gates):
        d.arrow((aff["r"][0], yy), g["l"])
    cb = d.box(565, 175 + OY, 130, 170, ["c(t) =", "f * c(t-1)", "+ i * g"], "model")
    for g, y in zip(gates[:3], ys[:3]):
        d.arrow(g["r"], (cb["l"][0], y))
    cin = d.oval(565, 62, 100, 36, ["c(t-1)"], "result")
    d.arrow(cin["b"], cb["t"], "kept by f", side="right")
    hb = d.box(565, 330 + OY, 150, 46, ["h(t) = o * tanh(c(t))"], "model")
    d.arrow(gates[3]["r"], (hb["l"][0], 330 + OY))
    cout = d.oval(745, 175 + OY, 70, 36, ["c(t)"], "result")
    d.arrow(cb["r"], cout["l"])
    d.path([(cb["r"][0] + 20, 175 + OY), (672, 175 + OY), (672, 285 + OY), (600, 285 + OY), (600, hb["t"][1])], radius=8)
    d.text(700, 240 + OY, ["tanh"], size=12, color="#4a5568")
    hout = d.oval(745, 330 + OY, 70, 40, ["h(t)"], "result")
    d.arrow(hb["r"], hout["l"])
    return d.svg()
