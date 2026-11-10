# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The text interface of each of brain's image pipelines: which encoder, which states of it, the sequence of rows that goes to the
# denoiser's reader and the pooled vector that joins the clock (`text_conditioning.interface_table` has the counts).
#
# Swedish Embedded AB implements solutions for text-conditioned image generators whose prompt interface is specified and checked.
# If your team needs expertise in text encoders for diffusion models, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import text_conditioning  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 940, 580
PX, EX, TX, SX = 62, 290, 520, 790          # prompt, encoder, states and sequence columns
EW, TW, SW = 184, 196, 232
ROWS = (118, 252, 386, 520)


def _numbers(name):
    for n, w, d, pooled, _, _ in text_conditioning.INTERFACES:
        if n == name:
            return f"{w} x {d:,}", f"{pooled:,}" if pooled else None
    raise KeyError(name)


def render() -> str:
    d = Diagram(W, H)
    for x, label in ((EX, "encoder"), (TX, "states read"), (SX, "rows, and the pooled vector")):
        d.text(x, 34, [label], size=13, color="#4a5568", weight="bold")
    prompt = d.oval(PX, (ROWS[0] + ROWS[-1]) / 2, 96, 64, ["prompt", "text"], "plain")
    s1, p1 = _numbers("SDXL")
    f1, q1 = _numbers("FLUX.1 dev")
    f2, _ = _numbers("FLUX.2 klein-4b")
    f2b, _ = _numbers("FLUX.2 klein-9b")
    z, _ = _numbers("Z-Image Turbo")
    spec = (
        (["CLIP-L and OpenCLIP-bigG,", "causal, 77 positions"], ["second-to-last block of each,", "not normalised, side by side"],
         [f"{s1} rows to the keys", "and values of 70 sites"], [f"{p1} bigG vector, projected,", "to the added embedding"]),
        (["T5-XXL, bidirectional,", "and CLIP-L, causal"], ["T5: last block, final norm", "CLIP-L: </s> row, normalised"],
         [f"{f1} rows to txt_in,", "then joint attention"], [f"{q1} CLIP-L vector, added", "to the timestep vector"]),
        (["Qwen3-4B or 8B, causal,", "27 of its 36 blocks"], ["after 9, 18 and 27 blocks,", "joined along the features"],
         [f"{f2} (4B) or {f2b.split(' x ')[1]} (9B)", "rows to txt_in, joint attention"], None),
        (["Qwen3-4B, causal,", "35 of its 36 blocks"], ["after 35 blocks, the second-", "to-last hidden state"],
         [f"{z} rows to cap_embedder,", "then joint attention"], None),
    )
    names = ("SDXL", "FLUX.1", "FLUX.2 klein", "Z-Image")
    mid = 140
    for y, name, (enc, tap, seq, pooled) in zip(ROWS, names, spec):
        e = d.box(EX, y, EW, 64, enc, "model")
        t = d.box(TX, y, TW, 64, tap, "accent")
        d.path([prompt["r"], (mid, prompt["r"][1]), (mid, y), e["l"]])
        d.arrow(e["r"], t["l"])
        if pooled:
            sq = d.box(SX, y - 28, SW, 48, seq, "result")
            pl = d.box(SX, y + 30, SW, 48, pooled, "result")
            d.path([t["r"], (t["r"][0] + 24, t["r"][1]), (t["r"][0] + 24, y - 28), sq["l"]])
            d.path([t["r"], (t["r"][0] + 24, t["r"][1]), (t["r"][0] + 24, y + 30), pl["l"]])
        else:
            sq = d.box(SX, y, SW, 48, seq, "result")
            d.arrow(t["r"], sq["l"])
        d.text(EX, y - 50, [name], size=12, color="#4a5568", weight="bold")
    return d.svg()
