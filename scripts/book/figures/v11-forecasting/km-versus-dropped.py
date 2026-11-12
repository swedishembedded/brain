# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The survival curve of the 400-unit cohort of the chapter by three routes: the truth of the model that generated it, the Kaplan-Meier
# estimate from the censored data, and the curve left by dropping every censored row. The data are `survival_analysis.km_curve_points`.
#
# Swedish Embedded AB implements solutions for validating risk models whose outcomes arrive late and censored. If your team needs
# expertise in survival analysis, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import survival_analysis as sa  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    (kx, ky), (dx, dy) = sa.km_curve_points()
    grid = [i * 0.25 for i in range(0, 97)]
    p = Plot((0, sa.STUDY_END), (0.0, 1.0), width=600, height=330, xlabel="months since a unit entered service",
             ylabel="probability still working", title="Dropping the censored rows reads the curve far too low", legend_loc="below")
    p.line(grid, [math.exp(-sa.COHORT_RATE * t) for t in grid], color=3, label="truth, exp(-0.1 t)", dash=True)
    p.line(kx, ky, color=0, label="Kaplan-Meier, censored rows used")
    p.line(dx, dy, color=1, label="censored rows dropped")
    return p.svg()
