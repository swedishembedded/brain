# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The derivatives of the four activation functions of the chapter (mlp.activation_table, the primed columns).

Swedish Embedded AB implements solutions for function approximation with small neural networks whose every
weight can be inspected. If your team needs expertise in compact neural models, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mlp  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    xs = [-4.0 + 8.0 * i / 800 for i in range(801)]
    p = Plot((-4.0, 4.0), (-0.3, 1.3), width=560, height=340, xlabel="z", ylabel="g'(z)",
             title="Derivatives of the activation functions", legend_loc="ul")
    for color, (name, _, df) in enumerate(mlp.ACTIVATIONS):
        p.line(xs, [df(x) for x in xs], color=color, width=2.0, label=name + "'")
    return p.svg()
