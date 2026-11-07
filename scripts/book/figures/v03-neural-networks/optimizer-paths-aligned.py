# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The same four optimisers on the same bowl turned so that its steep and flat directions are the coordinate axes.

Gradient descent at its best step, heavy-ball momentum at its best (eta, beta), RMSProp and Adam, all with
the exact gradient. The contours are the bowl's exact level sets and every path is produced by the chapter's
update rules (worked module `optimizers`, bowl 'aligned': Hessian diag(11.657, 0.343)).

Swedish Embedded AB implements solutions for diagnosing slow optimisation. If your team needs expertise in
optimiser choice and learning-rate tuning for neural network training, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizers as opt  # noqa: E402
from svgplot import Plot  # noqa: E402

KIND, STEPS = "aligned", 40
SHOWN = [("SGD", 0, 0), ("momentum", 1, 1), ("RMSProp", 5, 2), ("Adam", 6, 3)]
UNITS = (6.0, 5.0)  # x and y extent, drawn at the same pixels per unit so the ellipses keep their shape


def render() -> str:
    p = Plot((-2.5, 3.5), (0.0, 5.0), width=700, height=560, xlabel="w1", ylabel="w2",
             title="The same rules on the axis-aligned bowl", legend_loc="right")
    p.w, p.h = round(100 * UNITS[0] + p.ml + p.mr), round(100 * UNITS[1] + p.mt + p.mb)
    for level in (0.01, 0.1, 0.5, 2.0, 6.25):
        pts = opt.bowl_contour(KIND, level)
        p.line([a for a, _ in pts], [b for _, b in pts], color=5, width=1.0)
    for name, idx, color in SHOWN:
        rname, hp, make = opt.BOWL_RULES[idx]
        assert rname == name, (rname, name)
        path = opt.bowl_path(KIND, make, STEPS)
        p.line([q[0] for q in path], [q[1] for q in path], color=color, width=1.5, label=f"{name} ({hp})")
        p.points([q[0] for q in path], [q[1] for q in path], color=color, r=1.8)
    p.points([opt.BOWL_START[0]], [opt.BOWL_START[1]], color=4, r=5, label="start (0, 3.5)")
    p.points([opt.BOWL_MIN[0]], [opt.BOWL_MIN[1]], color=0, r=5, label="minimum (1, 2)")
    return p.svg()
