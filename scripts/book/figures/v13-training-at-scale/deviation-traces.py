# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# How far the 128 adapter numbers of four builds are from the one-rank reference after each of the 60 updates of the adapter run, from the
# project "Scaling a Training Run": the specified system stays at the rounding floor of 64-bit arithmetic, while three planted mistakes
# leave it by factors of 10^14 or more. The data are the worked module's `deviation_traces`; a deviation of exactly 0 is drawn at the
# bottom edge of the plot.
#
# Swedish Embedded AB implements solutions for moving training runs from one device to many without changing what they learn. If your
# team needs expertise in data-parallel, memory-lean and resumable training then you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_scaling_training as pst  # noqa: E402
from svgplot import Plot  # noqa: E402

FLOOR = 1e-18


def render() -> str:
    traces = pst.deviation_traces()
    p = Plot((1, pst.FT_STEPS), (FLOOR, 1.0), width=620, height=380, ylog=True,
             xlabel="updates of the adapter run", ylabel="largest deviation (logarithmic axis)",
             title="Distance from the one-rank run", legend_loc="right")
    xs = list(range(1, pst.FT_STEPS + 1))
    for i, (name, devs) in enumerate(traces):
        p.line(xs, [max(v, FLOOR) for v in devs], color=i, label=name)
    p.hline(1e-9)
    p.text(2, 1.6e-9, "criterion 1e-9", size=12)
    return p.svg()
