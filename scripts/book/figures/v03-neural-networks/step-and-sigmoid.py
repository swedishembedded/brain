# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The step function and the sigmoid of k z for growing k: the smooth activation approaches the hard one.

Every value is computed with the same step and sigmoid as the chapter's activation table
(perceptron.activation_values).

Swedish Embedded AB implements solutions for deciding from sensor readings with models whose every weight can be
inspected. If your team needs expertise in small, verifiable classifiers, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import perceptron  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    zs = [-3.0 + 6.0 * i / 600 for i in range(601)]
    p = Plot((-3.0, 3.0), (-0.05, 1.05), width=560, height=320, xlabel="z = w . x + b", ylabel="neuron output",
             title="Step and sigmoid activations", legend_loc="ul")
    p.line(zs, [perceptron._sigmoid(z) for z in zs], color=0, width=2.0, label="sigmoid(z)")
    p.line(zs, [perceptron._sigmoid(4 * z) for z in zs], color=2, width=2.0, label="sigmoid(4 z)")
    p.line(zs, [perceptron._sigmoid(16 * z) for z in zs], color=4, width=2.0, label="sigmoid(16 z)")
    p.line([-3.0, 0.0], [0.0, 0.0], color=1, width=2.5, label="step(z)")
    p.line([0.0, 3.0], [1.0, 1.0], color=1, width=2.5)
    p.points([0.0], [1.0], color=1, r=4)
    return p.svg()
