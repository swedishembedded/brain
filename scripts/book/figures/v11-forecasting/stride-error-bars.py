# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The true size of the error bar on a mean loss differential over a fixed stretch of 288 bars, against the error bar an evaluator would report
# by treating the origins as independent and by accounting for their overlap, as the stride between origins changes. The data are
# `evaluating_forecasts.stride_data`, the table of `stride_information`.
#
# Swedish Embedded AB implements solutions for forecast evaluation that separates skill from luck. If your team needs expertise in
# the statistics of forecast comparison, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import evaluating_forecasts as ef  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = ef.stride_data()
    strides = [r[0] for r in data]
    p = Plot((1, 12), (0.0, 0.25), width=560, height=320, xlabel="stride between origins (bars), horizon 12",
             ylabel="error bar on the mean difference", title="More origins, no more information", legend_loc="below")
    p.line(strides, [r[2] for r in data], color=0, label="true spread of the mean")
    p.line(strides, [r[3] for r in data], color=1, label="reported, origins treated as independent", dash=True)
    p.line(strides, [r[4] for r in data], color=2, label="reported, overlap accounted for")
    return p.svg()
