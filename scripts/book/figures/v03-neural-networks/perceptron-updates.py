# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The decision boundary of the perceptron after each of its updates on the six points of the chapter.

The weights are the ones the traced run produces (perceptron.perceptron_trace); the boundary after update k is the
line w1 x1 + w2 x2 + b = 0 for the weights after that update, so a number beside a line matches a row of the table.
The numbers under the lines are the update counts.

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

XLIM, YLIM = (-1.0, 5.0), (-1.5, 5.0)


def _line_ends(w, b):
    """Endpoints of w1 x1 + w2 x2 + b = 0 inside the plot box."""
    pts = []
    for x1 in XLIM:
        if w[1] != 0:
            y = -(b + w[0] * x1) / w[1]
            if YLIM[0] <= y <= YLIM[1]:
                pts.append((x1, y))
    for x2 in YLIM:
        if w[0] != 0:
            x = -(b + w[1] * x2) / w[0]
            if XLIM[0] <= x <= XLIM[1]:
                pts.append((x, x2))
    return sorted(set(pts))[:2]


def render() -> str:
    p = Plot(XLIM, YLIM, width=560, height=460, xlabel="x1", ylabel="x2",
             title="Perceptron boundary after each update", legend_loc="below")
    ups = perceptron._updates()
    last = len(ups)
    for k, (_, _, _, w, b) in enumerate(ups, start=1):
        ends = _line_ends(w, b)
        xs, ys = [e[0] for e in ends], [e[1] for e in ends]
        if k == last:
            p.line(xs, ys, color=0, width=3.0, label="final boundary (update %d)" % k)
        else:
            p.line(xs, ys, color="#a0aec0", width=1.5, dash=True, label="earlier boundaries" if k == 1 else "")
    pos = [(x1, x2) for x1, x2, y in perceptron.TRACE_DATA if y > 0]
    neg = [(x1, x2) for x1, x2, y in perceptron.TRACE_DATA if y < 0]
    p.points([a for a, _ in pos], [c for _, c in pos], color=2, label="label +1")
    p.points([a for a, _ in neg], [c for _, c in neg], color=3, label="label -1")
    for n, (x1, x2, _) in enumerate(perceptron.TRACE_DATA):
        p.text(x1 + 0.12, x2 + 0.2, f"P{n + 1}", size=11)
    for k, (_, _, _, w, b) in enumerate(ups, start=1):
        ends = _line_ends(w, b)
        ex, ey = min(ends, key=lambda e: e[1])  # the end on the lower edge of the box
        p.text(ex, YLIM[0] + 0.3, str(k), anchor="middle", size=11, color="#4a5568" if k != last else "#2b6cb0")
    return p.svg()
