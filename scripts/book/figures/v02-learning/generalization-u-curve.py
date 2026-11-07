# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Training error and fresh-data error of polynomial fits of rising degree to the same 20 rows.

The training error never rises as the degree grows; the fresh error falls to the
noise floor at the true degree and then turns up. Both are computed by the
chapter's worked module, on a log10 scale because the low degrees are far worse.

Swedish Embedded AB implements solutions for catching overfitting before a model
is deployed. If your team needs expertise in model evaluation for measurement
data, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import generalization as g  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = g._degree_sweep_data()
    degrees = [d for d, _, _ in data]
    p = Plot((-0.3, 9.3), (-0.1, 1.5), width=620, height=380, xlabel="polynomial degree d",
             ylabel="log10 of RMSE (deg C)", title="Training error keeps falling; fresh error turns up")
    p.hline(math.log10(g.NOISE_SD), color="#a0aec0")
    p.text(9.2, math.log10(g.NOISE_SD) + 0.04, "noise floor 1.2 deg C", anchor="end", size=11, color="#4a5568")
    p.line(degrees, [math.log10(t) for _, t, _ in data], color=0, width=2.4, label="training rows (20)")
    p.points(degrees, [math.log10(t) for _, t, _ in data], color=0, r=4)
    p.line(degrees, [math.log10(f) for _, _, f in data], color=1, width=2.4, label="fresh rows (2000)")
    p.points(degrees, [math.log10(f) for _, _, f in data], color=1, r=4)
    return p.svg()
