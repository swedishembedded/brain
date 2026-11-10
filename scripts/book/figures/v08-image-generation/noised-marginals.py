# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The density of the three clusters of data after t steps of the noising chain, in closed form (`diffusion.marginal_curves_data`): three
# separate bumps at t = 0, a flat shoulder where they merge, and a standard normal at t = 50.
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
    xs, curves = diffusion.marginal_curves_data()
    p = Plot((-6, 6), (0, 0.6), width=560, height=340, xlabel="x", ylabel="density p_t(x)", title="The data, noised for t steps", legend_loc="ur")
    for i, (t, ys) in enumerate(curves.items()):
        p.line(xs, [min(y, 0.6) for y in ys], color=i, label=f"t = {t}", width=2.0)
    return p.svg()
