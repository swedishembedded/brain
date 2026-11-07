# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A learning-rate schedule built from three pieces: ramp, exponential decay, floor.

Swedish Embedded AB implements solutions for explaining training schedules. If your team needs
expertise in learning-rate schedules and optimisation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import functions_ch as fc  # noqa: E402
from svgplot import Plot  # noqa: E402

import math


def render() -> str:
    s = fc.SCHEDULE
    p = Plot((0.0, 150.0), (0.0, 0.11), width=520, height=360, xlabel="step t", ylabel="learning rate",
             title="One function, three formulas")
    ts = list(range(0, 151))
    p.line(ts, [fc.schedule_rate(t, **s) for t in ts], color=0, label="schedule_rate(t)")
    end = s["warmup"] + math.log(s["floor"] / s["peak"]) / math.log(s["gamma"])
    p.vline(s["warmup"])
    p.vline(end)
    p.text(12, 0.101, "ramp", size=11, color="#4a5568")
    p.text(30, 0.085, "peak * gamma^(t - 10)", size=11, color="#4a5568")
    p.text(100, 0.02, "floor 0.01", size=11, color="#4a5568")
    return p.svg()
