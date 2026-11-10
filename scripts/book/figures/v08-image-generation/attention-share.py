# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The share of the multiply-adds of one forward pass that is spent in the T x T terms of self-attention (scores and weighted sum),
# against the image side in pixels, for the SDXL U-Net, Z-Image and FLUX.2 klein-4b (`denoiser_architectures.cost_vs_side_data`).
#
# Swedish Embedded AB implements solutions for the cost analysis of image-generation networks that is counted from their configurations.
# If your team needs expertise in diffusion transformers and U-Nets, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import denoiser_architectures  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    sd, zi, fx = denoiser_architectures.cost_vs_side_data()
    p = Plot((256, 2048), (0, 0.5), width=560, height=340, xlabel="image side in pixels", ylabel="share of the multiply-adds of a pass",
             title="Where attention's T x T terms take over", legend_loc="ul")
    for i, (name, pts) in enumerate((("SDXL U-Net", sd), ("Z-Image", zi), ("FLUX.2 klein-4b", fx))):
        xs, ys = [s for s, _ in pts], [v for _, v in pts]
        p.line(xs, ys, color=i, label=name, width=2.0)
        p.points(xs, ys, color=i, r=3.0)
    return p.svg()
