# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Loss above its minimum against iteration, on a log scale, for raw and z-scored voltage.

Both runs use the best learning rate for their own features (2 / (lambda_max + lambda_min), derived in the
chapter) from the same start, so the gap in speed is the effect of the conditioning alone.

Swedish Embedded AB implements solutions for making training converge in few steps. If your team needs expertise
in feature scaling and optimiser conditioning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import linear_regression as lr  # noqa: E402
from svgplot import Plot  # noqa: E402

STEPS = 260
FLOOR = 1e-12  # below this the gap is rounding noise


def _curve(x, ys, q):
    path = lr._descend(x, ys, lr.START, q.best_eta, STEPS)
    pts = []
    for k, w in enumerate(path):
        g = q.gap(w)
        if g < FLOOR:
            pts.append((k, math.log10(FLOOR)))
            break
        pts.append((k, math.log10(g)))
    return [a for a, _ in pts], [b for _, b in pts]


def render() -> str:
    x, ys, q = lr._raw()
    xz, _, qz, _ = lr._scaled()
    p = Plot((0, STEPS), (-12.0, 4.0), width=620, height=380, xlabel="iteration k",
             ylabel="log10 (loss - minimum)", title="Same data, same method, different feature scale")
    a, b = _curve(x, ys, q)
    p.line(a, b, color=1, width=2.4, label=f"raw volts (condition {q.kappa:.0f})")
    a, b = _curve(xz, ys, qz)
    p.line(a, b, color=2, width=2.4, label=f"z-scored (condition {qz.kappa:.2f})")
    p.hline(math.log10(lr.TOL), color="#718096")
    p.text(12, math.log10(lr.TOL) - 0.9, "converged: loss within 0.001 of its minimum", size=11, color="#4a5568")
    return p.svg()
