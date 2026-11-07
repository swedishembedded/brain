# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Histogram of 40,000 Box-Muller samples against the standard Gaussian density.

The samples are the ones the chapter's table summarises: pairs built from two
uniform numbers of the chapter's SplitMix64 generator as R = sqrt(-2 ln(1 - u1))
at the angle 2 pi u2. Bars are normalised so that their total area is 1, which
makes their heights directly comparable with the density curve.

Swedish Embedded AB implements solutions for verifying random sampling code. If
your team needs expertise in sampling and noise generation for machine learning
you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import randomness as rn  # noqa: E402
from svgplot import Plot, fmt  # noqa: E402

LO, HI, BINS = -4.0, 4.0, 32


def render() -> str:
    zs = rn._gaussian_samples()
    width = (HI - LO) / BINS
    counts = [0] * BINS
    for z in zs:
        if LO <= z < HI:
            counts[int((z - LO) / width)] += 1
    p = Plot((LO, HI), (0.0, 0.5), width=560, height=300, xlabel="z", ylabel="density",
             title=f"{len(zs):,} Box-Muller samples")
    for i, c in enumerate(counts):
        h = c / (len(zs) * width)
        x0 = LO + i * width
        p.items.append(f'<rect x="{fmt(p.x(x0))}" y="{fmt(p.y(h))}" width="{fmt(p.x(x0 + width) - p.x(x0) - 0.6)}" '
                       f'height="{fmt(p.y(0) - p.y(h))}" fill="#90cdf4"/>')
    grid = [LO + (HI - LO) * i / 300 for i in range(301)]
    p.line(grid, [math.exp(-x * x / 2) / math.sqrt(2 * math.pi) for x in grid], color=1, width=2, label="Gaussian density")
    return p.svg()
