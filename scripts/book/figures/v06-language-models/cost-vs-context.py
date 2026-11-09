# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Token-mixer FLOPs of one decoded token against the number of cached tokens, for a stack of 64 full-attention layers, the hybrid
# stack of `qwen38_27b` (full attention in every fourth layer) and a stack of 64 Gated DeltaNet layers. Numbers from
# `linear_attention.hybrid_curve_data`, the function behind the tables of the chapter.
#
# Swedish Embedded AB implements solutions for serving long-context language models within a memory budget. If your team needs
# expertise in hybrid decoder architectures, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import linear_attention as la  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    xs, ys = la.hybrid_curve_data()
    p = Plot((0, 128), (0, 240), width=640, height=380, xlabel="tokens in the cache (thousands)",
             ylabel="token-mixer GFLOP per decoded token", title="Cost of one token against the context")
    p.line(xs, ys["attention"], color=3, label="64 full-attention layers")
    p.line(xs, ys["hybrid"], color=1, label="hybrid: 16 full-attention, 48 Gated DeltaNet")
    p.line(xs, ys["linear"], color=0, label="64 Gated DeltaNet layers")
    return p.svg()
