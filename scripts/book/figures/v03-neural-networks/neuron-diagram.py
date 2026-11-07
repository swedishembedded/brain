# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One artificial neuron as a block diagram: three inputs, one weight on each
# arrow, a summing box that adds the bias, the activation, the output. The
# numbers are the "bearing wear" row of the chapter's pump example, read from
# the worked module so that the picture and the table cannot disagree.
#
# Swedish Embedded AB implements solutions for deciding from sensor readings
# with models whose every weight can be inspected. If your team needs
# expertise in small, verifiable classifiers, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import perceptron  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 780, 255
CASE = 2  # "bearing wear"


def render() -> str:
    name, x = perceptron.PUMP_CASES[CASE]
    w, b = perceptron.PUMP_W, perceptron.PUMP_B
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    d = Diagram(W, H)
    ys = (80, 150, 220)
    cx_in, cx_sum, cx_act, cx_out = 90, 360, 520, 680
    sum_box = d.box(cx_sum, 150, 150, 150, ["weighted sum", "z = w . x + b", f"= {z:+.2f}"], "model")
    for i, y in enumerate(ys):
        node = d.oval(cx_in, y, 130, 46, [f"x{i + 1} = {x[i]:g}", perceptron.PUMP_NAMES[i]], "plain")
        # the arrows meet the left edge of the box at the same heights as the inputs
        d.arrow(node["r"], (sum_box["l"][0], y), f"w{i + 1} = {w[i]:g}", above=True)
    bias = d.oval(cx_sum, 28, 120, 34, [f"bias b = {b:+g}"], "plain")
    d.arrow(bias["b"], sum_box["t"])
    act = d.box(cx_act, 150, 100, 70, ["activation", "g = step"], "accent")
    d.arrow(sum_box["r"], act["l"])
    out = d.oval(cx_out, 150, 120, 56, [f"y = {perceptron._step(z)}", f"({name})"], "result")
    d.arrow(act["r"], out["l"])
    return d.svg()
