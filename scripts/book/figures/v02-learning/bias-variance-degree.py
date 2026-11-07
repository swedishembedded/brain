# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bias squared, variance and total expected error of polynomial fits, by degree.

Computed by the chapter's simulation: 1000 training sets of 20 rows from the sensor
source. Degrees 0 and 6 leave the plotted range; their values are written on the plot.

Swedish Embedded AB implements solutions for choosing model capacity from
evidence. If your team needs expertise in bias-variance analysis for
measurement models, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import generalization as g  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    degrees = list(g.BV_DEGREES)
    stats = g._bv_simulation(g.BV_DEGREE_NAME, [(d, 0.0) for d in degrees])
    bias2 = [s[0] for s in stats]
    var = [s[1] for s in stats]
    total = [b + v + g.NOISE_SD ** 2 for b, v in zip(bias2, var)]
    p = Plot((-0.3, 6.3), (0.0, 20.0), width=620, height=380, xlabel="polynomial degree d",
             ylabel="squared error (deg C squared)", title="Bias squared, variance and their sum by degree")
    p.line(degrees, bias2, color=0, width=2.4, label="bias squared")
    p.line(degrees, var, color=1, width=2.4, label="variance")
    p.line(degrees, total, color=3, width=2.4, label="bias^2 + variance + noise")
    for series, color in ((bias2, 0), (var, 1), (total, 3)):
        p.points(degrees, series, color=color, r=3.5)
    p.text(1.5, 13.0, f"off the scale: degree 0 (bias^2 {bias2[0]:.0f}, total {total[0]:.0f})", size=11, color="#4a5568")
    p.text(1.5, 12.0, f"and degree 6 (variance {var[6]:.0f}, total {total[6]:.0f})", size=11, color="#4a5568")
    return p.svg()
