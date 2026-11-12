# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Which (query, key) pairs a frame block and a global block score, for three frames of four tokens each: the block-diagonal mask of the
# frame block and the full mask of the global block, with the counts of pairs from the chapter's arithmetic.
#
# Swedish Embedded AB implements solutions for attention that spans many views without exhausting the device. If your team needs
# expertise in multi-view transformers, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402

FRAMES, TOKENS = 3, 4


def render() -> str:
    n = FRAMES * TOKENS
    cell = 14
    side = n * cell
    panels = [("frame block", FRAMES * TOKENS * TOKENS, lambda i, j: i // TOKENS == j // TOKENS),
              ("global block", n * n, lambda i, j: True)]
    titles = [f"{name}: {pairs} pairs" for name, pairs, _ in panels]
    pw = max(side + 60, int(text_width(titles) + 40))
    ph = side + 96
    gap = 28
    W = int(2 * pw + gap + 40)
    H = int(ph + 40 + 30)
    d = Diagram(W, H)
    for p, (name, pairs, keep) in enumerate(panels):
        x0 = 20 + p * (pw + gap)
        y0 = 20
        d.panel(x0, y0, pw, ph, titles[p])
        gx = x0 + (pw - side) / 2
        gy = y0 + 40
        for i in range(n):
            for j in range(n):
                d.swatch(gx + j * cell, gy + i * cell, cell, cell, "result" if keep(i, j) else "plain")
        d.text(x0 + pw / 2, gy + side + 20, ["key (column), query (row)"], size=12, color="#4a5568")
        d.text(x0 + pw / 2, gy + side + 38, [f"{FRAMES} frames of {TOKENS} tokens"], size=12, color="#4a5568")
    return d.svg()
