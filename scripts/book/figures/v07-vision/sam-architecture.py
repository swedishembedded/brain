# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The data flow of a promptable segmentation model: an image encoder that runs once per image, a prompt encoder and a mask decoder that
# run once per prompt, with the counted FLOPs of the large SAM 2.1 model (`segment_anything.architecture_data`).
#
# Swedish Embedded AB implements solutions for interactive segmentation whose per-click cost is counted before it is paid. If your team
# needs expertise in promptable segmentation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import segment_anything as sa  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 940, 412


def render() -> str:
    data = sa.architecture_data()
    d = Diagram(W, H)
    d.panel(14, 14, 372, 372, "once per image")
    d.panel(402, 14, 524, 372, "once per prompt")
    image = d.oval(100, 120, 130, 56, ["image", "1024 x 1024 x 3"], "accent")
    enc = d.box(100, 250, 150, 96, ["image encoder", "Hiera trunk + neck", f"{data['trunk'] / 1e9:,.0f} billion FLOPs"])
    emb = d.box(300, 250, 150, 96, ["image embedding", "256 x 64 x 64", "+ 2 finer maps"], "result")
    prompts = d.oval(490, 120, 150, 56, ["prompts", "points, box, mask"], "accent")
    penc = d.box(490, 250, 150, 96, ["prompt encoder", "Fourier features", "+ type embeddings"])
    dec = d.box(690, 250, 160, 96, ["mask decoder", "two-way transformer", f"{data['decoder'] / 1e9:,.1f} billion FLOPs"])
    out = d.oval(860, 250, 110, 96, ["masks", "+ overlap", "per mask"], "accent")
    d.arrow(image["b"], enc["t"])
    d.arrow(enc["r"], emb["l"])
    d.arrow(prompts["b"], penc["t"])
    d.arrow(penc["r"], dec["l"], label="tokens")
    d.path([(emb["r"][0], emb["r"][1] + 30), (390, 330), (690, 330), (690, dec["b"][1])], radius=8, dash="5 3")
    d.text(640, 346, ["the stored embedding is read again for every prompt"], size=12, color=MUTED)
    d.arrow(dec["r"], out["l"])
    d.text(W / 2, H - 12, [f"{data['cache_bytes'] / 2 ** 20:.0f} MiB of features per image are kept in 32-bit numbers; "
                          f"one click costs {100 * data['decoder'] / data['trunk']:.2f} percent of the encoder's FLOPs."],
           size=12, color=MUTED)
    return d.svg()
