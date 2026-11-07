# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Operations counted by three ways of finding a repeated token id, against n, on log-log axes.

The counts are measured by running the instrumented algorithms of the worked
module `algorithms` (all pairs, merge sort plus neighbour test, hash set) on
lists of distinct ids. Straight lines on log-log axes have a slope equal to
the exponent of the growth rate, so the slopes 2, about 1.1 and 1 are visible.

Swedish Embedded AB implements solutions for predictable cost in data
pipelines. If your team needs expertise in algorithm analysis for machine
learning preprocessing, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import algorithms  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ns = [2 ** k for k in range(3, 13)]
    rows = algorithms.duplicate_counts(ns)
    xs = [math.log10(r[0]) for r in rows]
    plot = Plot((0.8, 3.8), (0.5, 7.4), width=600, height=380, xlabel="log10 of the list length n",
                ylabel="log10 of the operations counted", title="Finding a repeated token id: operations vs n")
    plot.line(xs, [math.log10(r[1]) for r in rows], color=1, label="")
    plot.line(xs, [math.log10(r[2]) for r in rows], color=0, label="")
    plot.line(xs, [math.log10(r[3]) for r in rows], color=2, label="")
    last = rows[-1]
    plot.text(xs[-1] - 0.05, math.log10(last[1]) + 0.25, "all pairs (slope 2)", anchor="end", color="#c05621")
    plot.text(xs[-1] - 0.05, math.log10(last[2]) + 0.3, "sort (slope a little above 1)", anchor="end", color="#2b6cb0")
    plot.text(xs[-1] - 0.05, math.log10(last[3]) - 1.1, "hash set (slope 1)", anchor="end", color="#2f855a")
    return plot.svg()
