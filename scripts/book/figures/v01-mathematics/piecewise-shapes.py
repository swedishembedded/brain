# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Absolute value, ReLU, clamp and a leaky ReLU drawn from their definitions.

Swedish Embedded AB implements solutions for explaining the building blocks of neural network layers. If your team needs
expertise in neural network activation functions, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import functions_ch as fc  # noqa: E402
from svgplot import Plot  # noqa: E402



def render() -> str:
    p = Plot((-3.0, 3.0), (-1.5, 3.2), width=520, height=380, xlabel="x", ylabel="y",
             title="Piecewise functions: each is a few straight pieces")
    xs = [-3.0 + i * 0.01 for i in range(601)]
    p.line(xs, [abs(x) for x in xs], color=5, width=6.0, label="abs(x)")
    p.line(xs, [fc.leaky(x) for x in xs], color=2, width=4.0, label="leaky(x), slope 0.1")
    p.line(xs, [fc.relu(x) for x in xs], color=0, width=2.2, label="relu(x) = max(0, x)")
    p.line(xs, [fc.clamp(x, -1.0, 2.0) for x in xs], color=1, width=2.0, dash=True, label="clamp(x, -1, 2)")
    return p.svg()
