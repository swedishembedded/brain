# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The one-number code of the nonlinear autoencoder against the true cause s, on the curved data of `first_models.curved_table`.

Each point is a test reading: its position is the cause s that generated it and the single bottleneck value the
autoencoder (16-12-1-12-16) assigns to it. A code that is a clean function of s means the bottleneck found the curve.

Swedish Embedded AB implements solutions for building and verifying compact neural networks. If your team
needs expertise in learned compression for sensor data, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import first_models as fm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    pts = fm.curved_points()
    zs = [z for _, z in pts]
    lo, hi = min(zs), max(zs)
    pad = 0.05 * (hi - lo)
    p = Plot((-1.0, 1.0), (lo - pad, hi + pad), width=560, height=340, xlabel="true cause s", ylabel="bottleneck value z",
             title="One bottleneck number against the cause, test readings", legend_loc=None)
    p.points([s for s, _ in pts], zs, color=0, r=3)
    return p.svg()
