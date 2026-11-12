# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The inverse-probability-of-censoring weighted Brier score of three predictors of the 300-subject cohort of the chapter at the horizons
# 1 .. 12 months. The data are `survival_analysis.brier_curves`.
#
# Swedish Embedded AB implements solutions for validating risk models whose outcomes arrive late and censored. If your team needs
# expertise in evaluating time-to-event predictions, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import survival_analysis as sa  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curves = sa.brier_curves()
    p = Plot((1, 12), (0.0, 0.3), width=600, height=330, xlabel="horizon t (months)", ylabel="Brier score at the horizon (lower is better)",
             title="A score belongs to a horizon", legend_loc="below")
    for i, (name, ys) in enumerate(curves):
        p.line(list(sa.SCORE_GRID), ys, color=i, label=name)
    return p.svg()
