# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The log signal-to-noise ratio ln(abar_t / (1 - abar_t)) against the fraction of the chain t / n, for brain's three schedules at
# 1000 steps and for the 50-step chain of the chapter (`diffusion.schedule_curves_data`). Where a curve ends is its terminal ratio.
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
    curves = diffusion.schedule_curves_data()
    p = Plot((0, 1), (-21, 12), width=560, height=340, xlabel="fraction of the chain t / n", ylabel="log signal-to-noise ratio",
             title="How fast each schedule destroys the signal", legend_loc="ll")
    p.hline(0.0)
    for i, (name, (xs, ys)) in enumerate(curves.items()):
        p.line(xs, ys, color=i, label=name, width=2.0, dash=(i == 3))
    return p.svg()
