# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The two hidden neurons of the hand-built XOR network as two lines, with the strip between them shaded.

h1 fires above x1 + x2 = 0.5 and h2 above x1 + x2 = 1.5 (perceptron.TWO_LAYER); the output neuron says
"h1 and not h2", which is the strip between the lines, and the two points with XOR = 1 are the ones inside it.

Swedish Embedded AB implements solutions for deciding from sensor readings with models whose every weight can be
inspected. If your team needs expertise in small, verifiable classifiers, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import perceptron  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402

LIM = (-0.5, 1.5)


def render() -> str:
    p = Plot(LIM, LIM, width=420, height=420, xlabel="x1", ylabel="x2", title="XOR with two hidden neurons",
             legend_loc="below")
    (w1, c1) = perceptron.TWO_LAYER["h1"]
    (w2, c2) = perceptron.TWO_LAYER["h2"]
    # lines x1 + x2 = -c / w (both hidden neurons have weights (1, 1))
    t1, t2 = -c1 / w1[0], -c2 / w2[0]
    # the strip t1 <= x1 + x2 < t2, clipped to the box, as a polygon
    corners = [(LIM[0], t1 - LIM[0]), (t1 - LIM[0], LIM[0]), (t2 - LIM[0], LIM[0]), (LIM[1], t2 - LIM[1]),
               (LIM[1], LIM[1]), (t2 - LIM[1], LIM[1]), (LIM[0], t2 - LIM[0])]
    # keep only vertices inside the box, in order around the strip
    poly = [(LIM[0], t1 - LIM[0]), (t1 - LIM[0], LIM[0]), (t2 - LIM[0], LIM[0]), (LIM[1], t2 - LIM[1]),
            (t2 - LIM[1], LIM[1]), (LIM[0], t2 - LIM[0])]
    del corners
    pts = " ".join(f"{fmt(p.x(a))},{fmt(p.y(b))}" for a, b in poly)
    p.items.append(f'<polygon points="{pts}" fill="#c6f6d5" fill-opacity="0.6" stroke="none"/>')
    xs = [LIM[0], LIM[1]]
    p.line(xs, [t1 - x for x in xs], color=0, width=2.0, label=f"h1 switches on at x1 + x2 = {t1:g}")
    p.line(xs, [t2 - x for x in xs], color=1, width=2.0, label=f"h2 switches on at x1 + x2 = {t2:g}")
    zero = [(x1, x2) for x1, x2, t in perceptron.XOR if t == 0]
    one = [(x1, x2) for x1, x2, t in perceptron.XOR if t == 1]
    p.points([a for a, _ in zero], [c for _, c in zero], color=3, r=6, label="XOR = 0")
    p.points([a for a, _ in one], [c for _, c in one], color=2, r=6, label="XOR = 1")
    return p.svg()
