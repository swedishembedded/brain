# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The noise the exact denoiser estimates from a noisy value, E[noise | x_t], against x_t at three steps (solid), and the trained
# network's estimate (dashed) (`diffusion.denoiser_curves_data`, the first run of the chapter's table).
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
    data = diffusion.denoiser_curves_data()
    p = Plot((-3.7, 3.7), (-3.4, 3.4), width=560, height=340, xlabel="noisy value x_t", ylabel="estimate of the noise",
             title="Exact and learned noise estimate", legend_loc="below")
    for i, (t, (xs, exact, net)) in enumerate(data.items()):
        p.line(xs, exact, color=i, label=f"exact, t = {t}", width=2.2)
        p.line(xs, net, color=i, label=f"network, t = {t}", width=1.6, dash=True)
    return p.svg()
