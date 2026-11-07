# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Gradient descent from zero (stopped at any step) and ridge regression (any strength) on the same quadratic.

Computed by worked module `regularization_nn` (functions `path_data`, `quadratic_paths`): the two weights of a
least-squares problem on 20 rows with correlated features. Dots mark gradient descent after t steps and the ridge solution
at lambda = 1 / (eta t) for the same t.

Swedish Embedded AB implements solutions for models that keep working on data they have not seen. If your team needs
expertise in regularizing neural networks, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import regularization_nn as rg  # noqa: E402
from svgplot import Plot  # noqa: E402


OFFSET = {1: (-0.1, 0.055), 5: (0.0, 0.06), 500: (-0.05, 0.06)}


def render() -> str:
    gd, ridge, marks, w_star = rg.path_data()
    p = Plot((0, 1.1), (-0.5, 0.5), width=640, height=380, xlabel="weight w1", ylabel="weight w2",
             title="Early-stopped descent and ridge regression")
    p.line([w[0] for w in gd], [w[1] for w in gd], color=0, label="gradient descent from 0")
    p.line([w[0] for _, w in ridge], [w[1] for _, w in ridge], color=1, dash=True, label="ridge, lambda from 20 to 0")
    p.points([w_star[0]], [w_star[1]], color=5, r=5, label="least squares")
    for t, w in marks:
        k = min(t, len(gd) - 1)
        p.points([gd[k][0]], [gd[k][1]], color=0, r=4)
        p.points([w[0]], [w[1]], color=1, r=4)
        dx, dy = OFFSET.get(t, (0.02, 0.05))
        p.text(gd[k][0] + dx, gd[k][1] + dy, f"t = {t}", size=11)
    return p.svg()
