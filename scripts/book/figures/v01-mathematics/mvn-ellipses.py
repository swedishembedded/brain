# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Samples from a two-dimensional Gaussian with the ellipses where (x-mu)^T Sigma^-1 (x-mu) = 1, 4, 9.

The points are drawn with the chapter's Cholesky-factor sampler from a bookrng
generator; the ellipse axes are the eigenvectors of Sigma with half-lengths
sqrt(c * lambda), the chapter's own formula.

Swedish Embedded AB implements solutions for explaining probabilistic models to
engineering teams. If your team needs expertise in multivariate noise models
and covariance estimation, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bookrng  # noqa: E402
import distributions as ds  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    mu = ds.MU
    pts = ds._sample_cholesky(bookrng.generator("distributions.mvn_figure"), mu, ds.SIGMA, 500)
    (l1, l2), (v1, v2) = ds._sym2_eigen(ds.SIGMA)
    p = Plot((-6.0, 8.0), (-5.0, 6.0), width=560, height=360, xlabel="x1", ylabel="x2",
             title="500 samples and the ellipses for c = 1, 4, 9")
    p.points([a[0] for a in pts], [a[1] for a in pts], color="#63b3ed", r=1.8)
    for c, color in ((1, "#c05621"), (4, "#2f855a"), (9, "#6b46c1")):
        ts = [2 * math.pi * i / 120 for i in range(121)]
        r1, r2 = math.sqrt(c * l1), math.sqrt(c * l2)
        xs = [mu[0] + r1 * math.cos(t) * v1[0] + r2 * math.sin(t) * v2[0] for t in ts]
        ys = [mu[1] + r1 * math.cos(t) * v1[1] + r2 * math.sin(t) * v2[1] for t in ts]
        p.line(xs, ys, color=color, width=2, label=f"c = {c}")
    p.arrow(mu[0], mu[1], mu[0] + 1.2 * v1[0], mu[1] + 1.2 * v1[1], color="#1a202c")
    p.arrow(mu[0], mu[1], mu[0] + 1.2 * v2[0], mu[1] + 1.2 * v2[1], color="#1a202c")
    p.text(mu[0] + 1.3 * v1[0] + 0.1, mu[1] + 1.3 * v1[1] - 0.45, "v1", size=12)
    p.text(mu[0] + 1.3 * v2[0] - 0.6, mu[1] + 1.3 * v2[1], "v2", size=12)
    return p.svg()
