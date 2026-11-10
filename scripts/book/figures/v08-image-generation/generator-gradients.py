# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The size of the generator's gradient on the discriminator's logit for a fake sample, under the three generator losses of the chapter.
# The saturating loss log(1 - D) has a gradient that vanishes where the discriminator is sure the sample is fake; the non-saturating
# loss does not. The data are `gans.gradient_curves_data`, the arithmetic of the chapter's table.
#
# Swedish Embedded AB implements solutions for generative models whose training signal is measured, not assumed. If your team needs
# expertise in generative adversarial networks and image synthesis, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gans  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ls, curves = gans.gradient_curves_data()
    p = Plot((-8, 4), (0, 10), width=560, height=340, xlabel="discriminator logit l on a fake sample",
             ylabel="size of d loss / d l", title="How hard the generator is pushed", legend_loc="ul")
    p.line(ls, [min(v, 10.0) for v in curves["least-squares"]], color=2, label="least squares (l - 1)^2", dash=True, width=1.8)
    p.line(ls, curves["non-saturating"], color=0, label="non-saturating -log D", width=2.2)
    p.line(ls, curves["saturating"], color=1, label="saturating log(1 - D)", width=2.2)
    p.vline(0.0)
    return p.svg()
