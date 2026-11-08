# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""SQNR of 4-bit symmetric codes against the group size, on heavy-tailed and on Gaussian weights, with the bits per
weight that the scales add.

The data comes from the worked module `quantization` (`symmetric_roundtrip` on its two datasets), the same arithmetic as
the table `quantization.group_size_tradeoff`. The horizontal axis is log2 of the group size G; a scale costs 16 bits.

Swedish Embedded AB implements solutions for storing machine learning models in few bits without losing the answers that
matter. If your team needs expertise in weight quantization and mixed-precision deployment, you can procure our services
by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import quantization as qz  # noqa: E402
from svgplot import Plot  # noqa: E402

GROUPS = (16384, 1024, 256, 128, 64, 32, 16, 8, 4)


def render() -> str:
    heavy, _ = qz.heavy_weights()
    gauss = qz.gaussian_weights("quantization.group_gauss", qz.HEAVY_N, qz.HEAVY_SIGMA)
    xs = [math.log2(g) for g in GROUPS]
    p = Plot((0.0, 16.0), (0.0, 28.0), width=600, height=360, xlabel="log2 of the group size G (one scale per G weights)",
             ylabel="SQNR in dB, 4-bit codes", title="Smaller groups buy accuracy and cost scale bits")
    for i, (name, data) in enumerate((("heavy-tailed weights", heavy), ("gaussian weights", gauss))):
        ys = [qz._sqnr_db(data, qz.symmetric_roundtrip(data, 4, g, 16)) for g in GROUPS]
        p.line(xs, ys, color=i, label=name, width=2.0)
        p.points(xs, ys, color=i, r=3.5)
    for g in (1024, 32, 8):
        p.text(math.log2(g), 1.8, f"{4 + 16 / g:.2f} bits", anchor="middle", size=11)
    return p.svg()
