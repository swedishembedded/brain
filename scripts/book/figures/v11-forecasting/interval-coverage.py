# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The share of 1000 series whose value lies inside a nominal 80 percent interval at each of the 24 horizons, for a constant-width,
# a model-based growing and a conformal interval. The data are `probabilistic_forecasting.coverage_curves`.
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
    curves = pf.coverage_curves()
    hs = list(range(1, 25))
    p = Plot((1, 24), (0.5, 1.0), width=560, height=320, xlabel="horizon h (bars after the origin)", ylabel="share of series inside",
             title="Coverage of nominal 80 percent intervals by horizon", legend_loc="below")
    for i, m in enumerate(pf.COVER_METHODS):
        p.line(hs, curves[m], color=i, label=m)
    p.hline(0.8)
    return p.svg()
