# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The key-value cache of a context of 32,768 tokens in bf16, against the number of key-value heads per layer, for the layer counts
# and head widths of Qwen3 0.6B and 8B. The dots mark the published choice, 8 key-value heads, and the multi-head extreme. Numbers
# from `modern_decoder.cache_vs_kv_heads_data`, the function behind the table of the chapter.
#
# Swedish Embedded AB implements solutions for serving language models within a memory budget. If your team needs expertise in
# key-value cache sizing, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import modern_decoder as md  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((0, 34), (0, 20), width=620, height=380, xlabel="key-value heads per layer",
             ylabel="GiB for 32,768 tokens", title="What a long context costs, by key-value heads", legend_loc="ul")
    for s, label, color in ((md.QWEN3_8B, "8B shape: 32 query heads, 36 layers", 3), (md.QWEN3_0_6B, "0.6B shape: 16 query heads, 28 layers", 0)):
        data = md.cache_vs_kv_heads_data(s)
        p.line([h for h, _, _ in data], [g for _, _, g in data], color=color, label=label, width=1.6)
        p.points([h for h, _, _ in data], [g for _, _, g in data], color=color, r=3.5)
        for h, _, g in data:
            if h in (8, s["H"]):
                if color == 3:
                    p.text(h - 0.6, g + 0.8, f"{g:.2f}", anchor="end", size=11)
                else:
                    p.text(h + 0.6, g - 1.2, f"{g:.2f}", size=11)
    return p.svg()
