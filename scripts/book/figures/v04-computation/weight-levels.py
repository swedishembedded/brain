# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A histogram of 4096 Gaussian weights with the 15 levels of the symmetric 4-bit code and the 16 levels of the affine one.

The weights, the scales and the levels come from the worked module `quantization` (`histogram_data`). Each level is drawn
as a vertical segment the height of the histogram's top, so the picture shows which levels the weights actually use:
the outer levels sit in the tail where there are almost no weights, and every weight between two neighbouring levels
is moved to the nearer one.

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

BINS = 48
LO, HI = -0.09, 0.09


def render() -> str:
    w, delta, sym, da, q0, aff = qz.histogram_data()
    width = (HI - LO) / BINS
    counts = [0] * BINS
    for x in w:
        b = int((x - LO) / width)
        if 0 <= b < BINS:
            counts[b] += 1
    top = max(counts)
    ymax = float(top) * 1.45
    p = Plot((LO, HI), (0.0, ymax), width=640, height=360, xlabel="weight value",
             ylabel="weights per bin", title="4-bit levels over a weight histogram")
    xs, ys = [LO], [0.0]
    for b, c in enumerate(counts):
        xs += [LO + b * width, LO + (b + 1) * width]
        ys += [float(c), float(c)]
    xs.append(HI)
    ys.append(0.0)
    p.line(xs, ys, color=5, label="histogram of the weights", width=1.5)
    for i, x in enumerate(sym):
        p.line([x, x], [0.0, top * 1.12], color=0, label="symmetric levels, 15 codes" if i == 0 else "", width=1.3)
    for i, x in enumerate(aff):
        p.line([x, x], [top * 1.18, top * 1.30], color=1, label="affine levels, 16 codes" if i == 0 else "", width=1.3)
    return p.svg()
