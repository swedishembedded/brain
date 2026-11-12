# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The share of test-block outcomes below the quantile forecast of each level, for the learned model's raw Gaussian head and after the
# recalibration fitted on the calibration block. The data are `project_forecaster.figure_reliability`.
#
# Swedish Embedded AB implements solutions for forecasts that state their own uncertainty and are judged honestly. If your team needs
# expertise in calibrating probabilistic forecasts, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_forecaster as pfc  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    levels, raw, recal = pfc.figure_reliability()
    p = Plot((0, 1), (0, 1), width=480, height=420, xlabel="quantile level", ylabel="share of test outcomes below the quantile",
             title="Quantile levels of the learned head, test block", legend_loc="lr")
    p.line([0, 1], [0, 1], color="#a0aec0", dash=True)
    p.line(levels, raw, color=0, label="raw Gaussian head")
    p.points(levels, raw, color=0)
    p.line(levels, recal, color=1, label="recalibrated on the calibration block")
    p.points(levels, recal, color=1)
    return p.svg()
