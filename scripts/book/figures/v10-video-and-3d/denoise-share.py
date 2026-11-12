# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The share of samples that end at the likelier of two futures (probability 0.8) against the number of denoiser evaluations, for a start
# of unit noise scaled to sigma_max and for unscaled unit noise (`world_models.diffusion_two_futures`).
#
# Swedish Embedded AB implements solutions for diffusion world models whose sampler is checked against an exact denoiser. If your team
# needs expertise in diffusion samplers and their step counts, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import world_models as wm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ns, scaled, unit = wm._share_plot_data()
    p = Plot((1, ns[-1]), (0.7, 1.02), width=560, height=320, xlabel="denoiser evaluations n", ylabel="share of samples at +1",
             title="Two futures, probabilities 0.8 and 0.2")
    p.hline(0.8)
    pts = [(n, s) for n, s in zip(ns, scaled) if s == s]
    p.line([a for a, _ in pts], [b for _, b in pts], color=0, width=2.0, label="start N(0, 25)")
    pts = [(n, s) for n, s in zip(ns, unit) if s == s]
    p.line([a for a, _ in pts], [b for _, b in pts], color=1, width=2.0, dash=True, label="start N(0, 1)")
    p.text(1.3, 0.785, "true share 0.8", anchor="start", size=11)
    return p.svg()
