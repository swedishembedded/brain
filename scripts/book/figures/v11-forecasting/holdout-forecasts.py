# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The last 60 bars of the chapter's series with four forecasts of the final 24 bars, all made from the first 96 bars only.
# The data are `time_series.sample_values` and the forecast functions of the baseline table.
#
# Swedish Embedded AB implements solutions for forecasting that is checked against honest baselines. If your team needs expertise
# in time-series evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time_series as ts  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    y = ts.sample_values()
    past = y[:ts.CUT]
    lo = 60
    p = Plot((lo, ts.CUT + ts.HORIZON - 1), (4, 21), width=600, height=340, xlabel="bar t", ylabel="value",
             title="Four forecasts of the last 24 bars, from the first 96", legend_loc="below")
    xs = list(range(lo, ts.CUT + ts.HORIZON))
    p.line(xs, y[lo:], color=5, label="the series", width=2.6)
    fx = list(range(ts.CUT, ts.CUT + ts.HORIZON))
    p.line(fx, ts._fc_naive(past, ts.HORIZON), color=3, label="naive", dash=True)
    p.line(fx, ts._fc_ma(past, ts.HORIZON), color=4, label="moving average", dash=True)
    p.line(fx, ts._fc_snaive_drift(past, ts.HORIZON), color=0, label="seasonal naive with drift")
    p.line(fx, ts._fc_ar(past, ts.HORIZON), color=1, label="AR(1) on seasonal differences")
    p.vline(ts.CUT - 0.5)
    p.text(ts.CUT - 1, 20.2, "past", anchor="end")
    p.text(ts.CUT, 20.2, "holdout")
    return p.svg()
