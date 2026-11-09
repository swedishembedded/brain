# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Bytes of key-value cache that one token adds, at the decoder shape of Qwen3 8B (36 layers, 32 query heads of width 128, bf16), for
# multi-head, grouped-query, multi-query and multi-head latent attention. The yellow bar is the one that compresses the keys and
# values themselves instead of sharing heads. Numbers from `attention_variants.cache_sizes_data`, the function behind the table of
# the chapter.
#
# Swedish Embedded AB implements solutions for serving language models within a memory budget. If your team needs expertise in
# key-value cache sizing, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import attention_variants as av  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 760, 300
LABEL_X, BAR_X, BAR_MAX = 20, 250, 400
ROWS = (92, 150, 208, 266)
BAR_H = 30
NOTES = ("32 key-value heads", "8 key-value heads", "1 key-value head", "latent 512 + rotary key 64")


def render() -> str:
    data = av.cache_sizes_data()
    top = max(b for _, b in data)
    d = Diagram(W, H)
    d.text(W / 2, 28, ["Cache one token adds, all 36 layers, bf16 (Qwen3 8B shape)"], size=14, weight="bold")
    d.text(BAR_X, 56, ["bar length is proportional to bytes per token"], size=12, color=MUTED, anchor="start")
    for y, (label, b), note in zip(ROWS, data, NOTES):
        d.text(LABEL_X, y - 8, [label], size=13, weight="bold", anchor="start")
        d.text(LABEL_X, y + 10, [note], size=12, color=MUTED, anchor="start")
        w = max(4.0, BAR_MAX * b / top)
        d.swatch(BAR_X, y - BAR_H / 2, w, BAR_H, "result" if "latent" in label else "model")
        d.text(BAR_X + w + 10, y, [f"{b / 1024:g} KiB"], size=13, anchor="start")
    return d.svg()
