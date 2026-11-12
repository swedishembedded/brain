# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The autocorrelation of the chapter's series at lags 1 to 14 before and after a seasonal difference, with the 95 percent band of
# white noise. The data are `time_series.acf_table`'s arithmetic (`_acf`).
#
# Swedish Embedded AB implements solutions for reading seasonality out of data before a model is chosen. If your team needs
# expertise in time-series diagnostics, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time_series as ts  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    y = ts.sample_values()
    kmax = 14
    raw = ts._acf(y, kmax)
    sd = ts._acf(ts._sdiff(y), kmax)
    p = Plot((0, 15), (-0.6, 1.0), width=560, height=320, xlabel="lag k (bars)", ylabel="autocorrelation r_k",
             title="Autocorrelation before and after a seasonal difference", legend_loc="ll")
    lags = list(range(1, kmax + 1))
    p.line(lags, raw[1:], color=0, label="the series")
    p.points(lags, raw[1:], color=0, r=3)
    p.line(lags, sd[1:], color=1, label="seasonal difference")
    p.points(lags, sd[1:], color=1, r=3)
    band = ts.BAND / math.sqrt(len(y))
    p.hline(band)
    p.hline(-band)
    p.hline(0.0, color="#4a5568", dash=False)
    return p.svg()
