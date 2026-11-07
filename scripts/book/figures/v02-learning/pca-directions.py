# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Two hundred correlated points and the two principal directions, each drawn as long as two standard deviations.

Swedish Embedded AB implements solutions for compressing and denoising high-dimensional sensor data. If your
team needs expertise in principal component analysis and low-rank models, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import classical as cl  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    pts = cl._cloud()
    mu, vals, vecs = cl._pca(pts)
    p = Plot((-4.0, 9.0), (-3.0, 8.0), width=520, height=436, xlabel="x1", ylabel="x2",
             title="Principal directions of a correlated cloud")
    p.points([q[0] for q in pts], [q[1] for q in pts], color="#90cdf4", r=3.0, label="200 points")
    p.points([mu[0]], [mu[1]], color="#1a202c", r=5, label="mean")
    for j, (lam, v) in enumerate(zip(vals, vecs)):
        L = 2.0 * math.sqrt(lam)
        p.arrow(mu[0], mu[1], mu[0] + L * v[0], mu[1] + L * v[1], color="#9b2c2c" if j == 0 else "#6b46c1")
        colour = "#9b2c2c" if j == 0 else "#6b46c1"
        p.text(-3.7, 6.8 - 0.8 * j, f"direction {j + 1} ({'red' if j == 0 else 'purple'}): {100 * lam / sum(vals):.1f}% of the variance", size=11, color=colour)
    return p.svg()
