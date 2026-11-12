# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The share of draws of N series from a panel of 200 for which a paired test finds that AR beats seasonal naive with drift, by N, when each series
# is judged on 1, 5 or 15 origins. The data are `evaluating_forecasts.power_data`, the table of `how_many_series`.
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
    data = ef.power_data()
    p = Plot((ef.POWER_SIZES[0], ef.POWER_SIZES[-1]), (0.0, 1.0), width=560, height=320, xlabel="number of series N in the panel",
             ylabel="share of draws that find AR better", title="How many series before a 0.023 edge in error shows", legend_loc="below")
    for i, (name, _, power) in enumerate(data):
        p.line(list(ef.POWER_SIZES), power, color=i, label=name.split(" (")[0])
    p.hline(0.8)
    return p.svg()
