# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Cache memory of one sequence against the context length, for several configurations.

The 7.5-billion and the 70-billion decoder of Volume V, Chapter 12 (8 key-value heads) in bf16 and in int8 (1 + 4/128 bytes per element),
and the 80-layer decoder of Volume VI, Chapter 3 with four windowed layers to each global one, int8 (`long_context.cache_curve_data`).

Swedish Embedded AB implements solutions for serving long contexts in the least device memory that keeps accuracy. If your team
needs expertise in key-value cache sizing, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import long_context as lc  # noqa: E402
from svgplot import Plot  # noqa: E402

GIB = 2 ** 30


def render() -> str:
    ctxs = [1024 * 2 ** k for k in range(0, 8)]          # 1,024 .. 131,072
    curves = lc.cache_curve_data(ctxs)
    p = Plot((1024, 131072), (0.03, 100), width=620, height=380, xlabel="tokens in the sequence", ylabel="cache of one sequence (GiB)",
             title="Cache memory against context", xlog=True, ylog=True, legend_loc="ul")
    colors = {"7.5B, bf16": 0, "7.5B, int8": 2, "70B, bf16": 1, "70B, int8": 3, "80 layers, 4:1 windowed, int8": 4}
    for name, ys in curves.items():
        p.line(ctxs, [y / GIB for y in ys], color=colors[name], label=name, dash=("int8" in name))
    return p.svg()
