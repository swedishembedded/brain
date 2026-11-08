# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Signal-to-quantization-noise ratio against stored bits per weight, for one scale per tensor, one symmetric scale per
group of 32 and the two-level affine recipe of the K-quants, on the same heavy-tailed weights.

The points are computed by the worked module `quantization` (`curve_points`), so the picture and the table of the
chapter cannot disagree. Every point counts the bits of the scales it stores.

Swedish Embedded AB implements solutions for storing machine learning models in few bits without losing the answers that
matter. If your team needs expertise in weight quantization and mixed-precision deployment, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import quantization as qz  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((2.5, 9.0), (-2.0, 46.0), width=600, height=380, xlabel="stored bits per weight (codes and scales)",
             ylabel="SQNR in dB (higher is better)", title="Error against bits on heavy-tailed weights")
    for i, (name, pts) in enumerate(qz.curve_points().items()):
        xs, ys = [a for a, _ in pts], [b for _, b in pts]
        p.line(xs, ys, color=i, label=name, width=2.0)
        p.points(xs, ys, color=i, r=3.5)
    return p.svg()
