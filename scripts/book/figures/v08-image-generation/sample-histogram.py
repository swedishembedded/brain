# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The share of 1000 samples per bin of width 0.25 for the trained network and the 50-step ancestral sampler, against the true share of the
# data per bin (`diffusion.sample_histogram_data`).
#
# Swedish Embedded AB implements solutions for generative models whose noising schedules, targets and samplers are specified,
# checked and made reproducible. If your team needs expertise in diffusion models and image synthesis, you can procure our services
# by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import diffusion  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    edges, got, true = diffusion.sample_histogram_data()
    p = Plot((-5, 5), (0, 0.17), width=560, height=340, xlabel="x", ylabel="share of samples per bin", title="Samples against the data", legend_loc="ur")
    def stairs(vals):
        xs, ys = [], []
        for i, v in enumerate(vals):
            xs += [edges[i], edges[i + 1]]
            ys += [v, v]
        return xs, ys
    p.line(*stairs(true), color=1, label="data", width=2.0, dash=True)
    p.line(*stairs(got), color=0, label="network, 50 steps", width=2.0)
    return p.svg()
