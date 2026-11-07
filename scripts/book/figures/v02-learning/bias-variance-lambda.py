# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bias squared, variance and total expected error of a degree-9 model as the penalty grows.

Computed by the chapter's simulation: 1000 training sets of 20 rows, a ridge fit with
each penalty lambda. Both axes are log10 because the values span five orders of magnitude.

Swedish Embedded AB implements solutions for tuning regularization against
held-out evidence. If your team needs expertise in the bias-variance
tradeoff for measurement models, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import generalization as g  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    lams = list(g.BV_LAMBDAS)
    stats = g._bv_simulation(g.BV_RIDGE_NAME, [(g.MAX_DEGREE, lam) for lam in lams])
    xs = [math.log10(lam) for lam in lams]
    bias2 = [math.log10(s[0]) for s in stats]
    var = [math.log10(s[1]) for s in stats]
    total = [math.log10(s[0] + s[1] + g.NOISE_SD ** 2) for s in stats]
    p = Plot((-3.3, 2.3), (-1.8, 4.6), width=620, height=380, xlabel="log10 of the penalty lambda",
             ylabel="log10 of squared error", title="Degree 9: bias and variance as the penalty grows")
    p.line(xs, bias2, color=0, width=2.4, label="bias squared")
    p.line(xs, var, color=1, width=2.4, label="variance")
    p.line(xs, total, color=3, width=2.4, label="bias^2 + variance + noise")
    for series, color in ((bias2, 0), (var, 1), (total, 3)):
        p.points(xs, series, color=color, r=3.5)
    return p.svg()
