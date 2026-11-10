# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Histograms (bin width 0.25) of the end points of 1000 samples of class 1 under classifier-free guidance with scale 0, 1, 3 and 8,
# against the density of the class itself (`samplers_and_guidance.guidance_curves_data`).
#
# Swedish Embedded AB implements solutions for prompt adherence and guidance of generative models that are measured against exact
# references. If your team needs expertise in classifier-free guidance, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import samplers_and_guidance  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    mids, curves, cls = samplers_and_guidance.guidance_curves_data()
    p = Plot((-2.5, 5.0), (0, 1.5), width=560, height=340, xlabel="end point x_0", ylabel="share of samples per unit",
             title="Samples of class 1 at four guidance scales", legend_loc="ur")
    p.line(mids, cls, color="#a0aec0", dash=True, width=1.6, label="the class itself")
    for i, (g, ys) in enumerate(curves.items()):
        p.line(mids, ys, color=i, label=f"g = {g:g}", width=2.0)
    return p.svg()
