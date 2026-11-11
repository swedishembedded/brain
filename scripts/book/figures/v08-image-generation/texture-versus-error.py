# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The error of a restorer against the amplitude of the texture it produces, in units of the real texture (`restoration.feature_weight_rows`):
# 1 + a^2 for every restorer that returns a texture of amplitude a, with the squared-error optimum at a = 0 and a sampler at a = 1.
#
# Swedish Embedded AB implements solutions for restoration whose price in distortion is stated. If your team needs expertise in the
# trade between fidelity and realism in restored pictures, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import restoration  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((0, 1.15), (0.9, 2.2), width=560, height=340, xlabel="amplitude of the produced texture a (real texture = 1)",
             ylabel="error per pixel (units of the real texture squared)", title="What each unit of texture costs", legend_loc="ul")
    xs = [i / 100 for i in range(0, 116)]
    p.line(xs, [1 + a * a for a in xs], color="#a0aec0", width=1.6, label="1 + a squared")
    rows = restoration.feature_weight_rows((0.0, 0.25, 1.0, 4.0, 16.0))
    p.points([a for _, a, _ in rows], [e for _, _, e in rows], color=0, label="squared error plus a feature weight", r=4.5)
    p.points([1.0], [2.0], color=1, label="a sample (the sampler of the toy)", r=5.5)
    p.points([0.0], [1.0], color=2, label="the mean (squared-error optimum)", r=5.5)
    return p.svg()
