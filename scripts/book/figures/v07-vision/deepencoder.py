# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The vision half of the two document readers of the chapter, with the token counts of each stage computed from the grids of
# `ocr.grids_data`: SAM and a 16x convolutional compressor, then either CLIP-L over the 256 tokens (first generation) or a stack
# of learned queries under a prefix mask (second generation).
#
# Swedish Embedded AB implements solutions for document-reading models whose token budget is counted before it is paid. If your team
# needs expertise in vision encoders for OCR, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ocr  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 490
BOX_W, BOX_H = 136, 96
XS = [14 + 18 + BOX_W / 2 + i * (BOX_W + 24) for i in range(6)]


def render() -> str:
    g = {name: (n, n2) for name, _, n, _, n2, _, _, _ in ocr.grids_data()}
    n, n2 = g["global view"]
    patches, tokens = n * n, n2 * n2
    wins = len(ocr.SAM["global_blocks"])
    d = Diagram(W, H)
    d.panel(14, 14, 972, 208, "first generation: SAM, a 16x compressor, CLIP-L")
    d.panel(14, 240, 972, 208, "second generation: SAM, a 16x compressor, 256 learned queries")
    for top, v2 in ((118, False), (344, True)):
        y = top + 12
        page = d.box(XS[0], y, BOX_W, BOX_H, ["page view", f"{ocr.GLOBAL_SIDE} x {ocr.GLOBAL_SIDE}", "pixels"], "accent")
        sam = d.box(XS[1], y, BOX_W, BOX_H, ["SAM ViT-B", f"{patches:,} patches", f"{ocr.SAM['layers'] - wins} windowed,", f"{wins} global blocks"])
        width = ocr.V2["sam_out"] if v2 else ocr.SAM["out"]
        comp = d.box(XS[2], y, BOX_W, BOX_H, ["compressor", "2 convs, stride 2", f"{tokens} tokens", f"of width {width}"])
        if v2:
            mid = d.box(XS[3], y, BOX_W, BOX_H, ["24 blocks, mask", f"{tokens} tokens +", f"{ocr.V2['q_global']} queries", f"= {2 * tokens} rows"])
            proj = d.box(XS[4], y, BOX_W, BOX_H, ["keep the queries,", f"Linear {ocr.V2['d']} to", f"{ocr.DEC['D']}", f"{tokens} rows"])
            out = d.box(XS[5], y, BOX_W, BOX_H, ["image block", f"{ocr.V2['q_global'] + 1} rows:", f"{tokens} + 1 separator"], "result")
        else:
            mid = d.box(XS[3], y, BOX_W, BOX_H, ["CLIP-L", f"{tokens + 1} rows", f"{ocr.CLIP['layers']} global blocks", f"width {ocr.CLIP['d']}"])
            proj = d.box(XS[4], y, BOX_W, BOX_H, ["concat, Linear", f"{ocr.CLIP['d'] + ocr.SAM['out']} to {ocr.DEC['D']}", f"{tokens} rows"])
            out = d.box(XS[5], y, BOX_W, BOX_H, ["image block", f"{tokens} + {n2} newline", "+ 1 separator", f"= {ocr.row_count(n2, 0, 0, 0)} rows"], "result")
        d.arrow(page["r"], sam["l"])
        d.arrow(sam["r"], comp["l"])
        d.arrow(comp["r"], mid["l"])
        d.arrow(mid["r"], proj["l"])
        d.arrow(proj["r"], out["l"])
        if not v2:
            d.path([comp["t"], (comp["t"][0], y - BOX_H / 2 - 22), (proj["t"][0], y - BOX_H / 2 - 22), proj["t"]], radius=8)
            d.text((comp["t"][0] + proj["t"][0]) / 2, y - BOX_H / 2 - 32, [f"the {tokens} compressor tokens also go to the concat"], size=12, color=MUTED)
    d.text(W / 2, H - 18, ["Both generations compress the 4,096 patches of SAM by 16 before the global stage, and feed the same 1280-wide decoder."], size=12, color=MUTED)
    return d.svg()
