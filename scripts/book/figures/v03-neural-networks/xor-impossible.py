# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The four XOR points and three candidate single-neuron boundaries, none of which gets all four right.

Each candidate is a line w1 x1 + w2 x2 + b = 0 with the side of w . x + b >= 0 called 1. The count in the legend is
the number of the four points that line classifies correctly, computed from the points; the best any line can do is 3.

Swedish Embedded AB implements solutions for deciding from sensor readings with models whose every weight can be
inspected. If your team needs expertise in small, verifiable classifiers, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import perceptron  # noqa: E402
from svgplot import Plot  # noqa: E402

# (w1, w2, b, colour, name): the first is the OR neuron, the second the AND neuron, the third a vertical cut.
CANDIDATES = ((1.0, 1.0, -0.5, 0, "x1 + x2 - 0.5 = 0"), (1.0, 1.0, -1.5, 1, "x1 + x2 - 1.5 = 0"), (1.0, 0.0, -0.5, 4, "x1 - 0.5 = 0"))
LIM = (-0.5, 1.5)


def _correct(w1, w2, b):
    return sum(perceptron._step(w1 * x1 + w2 * x2 + b) == t for x1, x2, t in perceptron.XOR)


def _segment(w1, w2, b):
    """Endpoints of the line inside the box LIM x LIM."""
    if w2 != 0:
        return [LIM[0], LIM[1]], [-(b + w1 * x) / w2 for x in LIM]
    x = -b / w1
    return [x, x], [LIM[0], LIM[1]]


def render() -> str:
    p = Plot(LIM, LIM, width=420, height=420, xlabel="x1", ylabel="x2", title="XOR: no line gets all four points",
             legend_loc="below")
    for w1, w2, b, c, name in CANDIDATES:
        xs, ys = _segment(w1, w2, b)
        p.line(xs, ys, color=c, width=2.0, label=f"{name}: {_correct(w1, w2, b)} of 4 right")
    zero = [(x1, x2) for x1, x2, t in perceptron.XOR if t == 0]
    one = [(x1, x2) for x1, x2, t in perceptron.XOR if t == 1]
    p.points([a for a, _ in zero], [c for _, c in zero], color=3, r=6, label="XOR = 0")
    p.points([a for a, _ in one], [c for _, c in one], color=2, r=6, label="XOR = 1")
    return p.svg()
