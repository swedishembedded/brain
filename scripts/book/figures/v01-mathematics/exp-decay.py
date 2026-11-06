# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Exponential learning-rate decay lr0 * gamma^t for three decay factors.

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
    lr0 = 0.1
    p = Plot((0.0, 240.0), (0.0, 0.1), width=520, height=360, xlabel="step t", ylabel="learning rate",
             title="Exponential decay: lr0 * gamma^t, lr0 = 0.1")
    ts = list(range(0, 241))
    for k, g in enumerate((0.99, 0.97, 0.94)):
        p.line(ts, [lr0 * g ** t for t in ts], color=k, label=f"gamma = {g}")
    p.hline(0.01)
    p.text(2, 0.0115, "ten times lower: 0.01", size=11, color="#4a5568")
    for k, g in enumerate((0.99, 0.97, 0.94)):
        t10 = math.log(10) / -math.log(g)
        p.points([t10], [0.01], color=k)
    return p.svg()
