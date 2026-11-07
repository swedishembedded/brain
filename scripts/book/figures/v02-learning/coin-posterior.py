# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A Beta(5, 5) prior, the likelihood of 7 heads in 10 flips, and the posterior, each scaled to unit area.

The posterior peaks between the prior's centre (0.5) and the likelihood's (0.7), at 0.611, the maximum a posteriori
estimate of the chapter (11 of 18 in the posterior counts). The curves are computed on a grid of 999 points from
theta^(a-1) (1 - theta)^(b-1) and the likelihood, then normalised by the trapezoid rule.

Swedish Embedded AB implements solutions for combining prior knowledge with measurements. If your team needs
expertise in Bayesian estimation, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import likelihood as lk  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    a = b = 5
    xs, prior = lk.beta_posterior_curve(a, b, 0, 0)
    _, lik = lk.beta_posterior_curve(1, 1, lk.HEADS, lk.TAILS)
    _, post = lk.beta_posterior_curve(a, b, lk.HEADS, lk.TAILS)
    peak = max(range(len(xs)), key=lambda i: post[i])
    p = Plot((0, 1), (0, 4.4), width=620, height=330, xlabel="coin bias theta", ylabel="density (area 1)")
    p.line(xs, prior, color=5, width=2.4, dash=True, label="prior Beta(5, 5)")
    p.line(xs, lik, color=1, width=2.4, label="likelihood, 7 heads in 10")
    p.line(xs, post, color=0, width=3.0, label="posterior")
    p.vline(xs[peak], color="#2b6cb0")
    p.text(xs[peak] - 0.01, 4.0, f"MAP {xs[peak]:.3f}", anchor="end", size=11)
    return p.svg()
