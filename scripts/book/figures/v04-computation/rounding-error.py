# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The largest relative rounding error against the size of the number, for fp16, bf16, E4M3 and E5M2.

For every pair of neighbouring representable numbers the error is largest at their midpoint, where
rounding errs by half the gap; the worked module `number_formats` takes the largest such error in each
binade (each factor of two). A flat line is a format with a constant number of significant bits; the rise at the
left is the subnormal range, where the gap stops shrinking.

Swedish Embedded AB implements solutions for numerically sound low-precision machine learning. If your
team needs expertise in rounding error of narrow number formats, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import number_formats as nf  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((-42.0, 42.0), (-3.8, 0.3), width=600, height=360, xlabel="log10 of the number being rounded",
             ylabel="log10 of the largest relative error", title="Worst relative rounding error by magnitude")
    for i, f in enumerate(["bf16", "fp16", "E5M2", "E4M3"]):
        env = nf.error_envelope(f)
        p.line([x for x, _ in env], [math.log10(e) for _, e in env], color=i, label=f)
    return p.svg()
