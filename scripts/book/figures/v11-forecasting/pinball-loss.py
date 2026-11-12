# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The mean pinball loss of the nine values of the chapter as a function of the forecast q at three levels, with the sample quantile
# that minimises each marked. The data are `probabilistic_forecasting.pinball_curve`.
#
# Swedish Embedded AB implements solutions for forecasts that state their own uncertainty. If your team needs expertise in
# probabilistic forecasting and its calibration, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probabilistic_forecasting as pf  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    qs = [k / 4 for k in range(0, 4 * 22 + 1)]
    p = Plot((0, 22), (0, 10), width=560, height=320, xlabel="forecast q", ylabel="mean pinball loss",
             title="Mean pinball loss of the nine values against q", legend_loc="below")
    colors = {0.1: 0, 0.5: 1, 0.9: 2}
    for tau in pf.SAMPLE9_LEVELS:
        p.line(qs, pf.pinball_curve(tau, qs), color=colors[tau], label=f"tau = {tau}")
    srt = sorted(pf.SAMPLE9)
    for tau in pf.SAMPLE9_LEVELS:
        q = pf._inverse_cdf_quantile(srt, tau)
        p.points([q], pf.pinball_curve(tau, [q]), color=colors[tau], r=5)
    return p.svg()
