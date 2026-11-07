# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The minimiser of one weight under an L2 and an L1 penalty, as a function of the unpenalised value a.

With lambda = 1 the L2 penalty halves every weight (a / (1 + lambda)), a straight line that never touches zero except
at a = 0. The L1 penalty moves every weight toward zero by lambda and holds it at exactly zero while |a| <= lambda
(soft-thresholding). Both curves are the formulas the chapter derives, from the worked module likelihood.

Swedish Embedded AB implements solutions for models that keep only the inputs that matter. If your team needs expertise
in sparse and regularized estimation, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import likelihood as lk  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    lam = 1.0
    a = [-4 + 0.05 * i for i in range(161)]
    p = Plot((-4, 4), (-4, 4), width=480, height=420, xlabel="unpenalised weight a", ylabel="penalised weight")
    p.hline(0.0, dash=False)
    p.line(a, a, color=5, width=1.6, dash=True, label="no penalty")
    p.line(a, [v / (1 + lam) for v in a], color=0, width=2.8, label="L2: a / (1 + lambda)")
    p.line(a, [lk.soft_threshold(v, lam) for v in a], color=1, width=2.8, label="L1: soft threshold")
    return p.svg()
