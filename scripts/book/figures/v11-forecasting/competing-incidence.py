# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The cumulative incidence of cause a in the two-cause cohort of the chapter: the truth, the Aalen-Johansen estimate and one minus the
# Kaplan-Meier estimate that treats cause b as a censoring. The data are `survival_analysis.competing_curve_points`.
#
# Swedish Embedded AB implements solutions for validating risk models whose outcomes arrive late, censored and competing. If your team
# needs expertise in competing-risks prediction, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import survival_analysis as sa  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    grid, truth, aj, naive = sa.competing_curve_points()
    p = Plot((0, 24), (0.0, 1.0), width=600, height=330, xlabel="months", ylabel="probability of cause a by that time",
             title="One minus Kaplan-Meier overstates a cause that has competitors", legend_loc="below")
    p.line(grid, truth, color=3, label="truth", dash=True)
    p.line(grid, aj, color=0, label="Aalen-Johansen")
    p.line(grid, naive, color=1, label="1 - Kaplan-Meier, cause b as censoring")
    return p.svg()
