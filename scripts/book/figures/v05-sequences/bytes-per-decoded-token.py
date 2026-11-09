# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bytes read to decode one token, against the number of tokens already in the cache.

A decoder with 32 layers and 128-wide heads (`kv_cache.DECODER_8B`, 7.5 billion parameters in matrix products) stored in bf16 reads
all its weights for every token (the dashed horizontal line) and the keys and values of every earlier token in every layer, which
grow with the context: 131,072 bytes per token with 8 key-value heads and 524,288 with 32 (`kv_cache.cache_bytes_per_token`). The
dots mark where the cache read equals the weights read.

Swedish Embedded AB implements solutions for serving language models without recomputing the past. If your team needs expertise
in key-value caches and decode performance, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import kv_cache as kc  # noqa: E402
from svgplot import Plot  # noqa: E402

CTX_MAX = 131072


def render() -> str:
    p = Plot((0, CTX_MAX), (0, 90), width=620, height=380, xlabel="tokens already in the cache",
             ylabel="GB read per decoded token", title="What one decoded token reads", legend_loc="ul")
    ctxs = [0, CTX_MAX]
    for s, label, color in ((kc.DECODER_8B_MHA, "weights + cache, 32 key-value heads", 3), (kc.DECODER_8B, "weights + cache, 8 key-value heads", 0)):
        data = kc.bytes_split_data(s, ctxs)
        p.line(ctxs, [w + c for w, c in data], color=color, label=label)
    w = kc.weight_params(kc.DECODER_8B) * kc.BYTES / 1e9
    p.line(ctxs, [w, w], color=5, dash=True, width=1.4, label="weights alone (8 key-value heads)")
    for s, color in ((kc.DECODER_8B_MHA, 3), (kc.DECODER_8B, 0)):
        cross = kc.weight_params(s) * kc.BYTES / kc.cache_bytes_per_token(s)
        y = 2 * kc.weight_params(s) * kc.BYTES / 1e9
        p.points([cross], [y], color=color, r=4.5)
        p.text(cross, y - 6, f"{cross:,.0f}", anchor="middle", size=11)
    return p.svg()
