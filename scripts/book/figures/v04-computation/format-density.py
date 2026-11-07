# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""How many representable numbers each 16-bit and 8-bit format has in every power of ten.

Every finite positive code of fp16, bf16, E4M3 and E5M2 is decoded by the worked module `number_formats`
and counted by the decade [10^k, 10^(k+1)) it falls in, so the plateau height is the format's precision
(2^n_m values per factor 2) and the plateau length is its range.

Swedish Embedded AB implements solutions for choosing number formats for machine learning workloads. If
your team needs expertise in range, precision and scaling of low-precision formats, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import number_formats as nf  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((-42.0, 42.0), (-0.2, 4.0), width=600, height=360, xlabel="decade k: values v with 10^k <= v < 10^(k+1)",
             ylabel="log10 of the number of values", title="Representable positive numbers per power of ten")
    for i, f in enumerate(["bf16", "fp16", "E5M2", "E4M3"]):
        dens = nf.density_by_decade(f)
        ks = sorted(dens)
        p.line([k + 0.5 for k in ks], [math.log10(dens[k]) for k in ks], color=i, label=f)
    return p.svg()
