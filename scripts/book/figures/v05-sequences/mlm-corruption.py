# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The masking procedure on one sentence of the chapter's barnyard language: the original tokens, which positions were selected and
# how (`[MASK]`, a random token, kept), the inputs the model reads, and the positions that carry loss. The sentence and its draws are
# those of `encoders.corruption_example_data`, the first sentence of the stream in which all three outcomes occur.
#
# Swedish Embedded AB implements solutions for building masked-language-model training data. If your team needs expertise in
# pre-training encoders, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import encoders  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, RED, Diagram  # noqa: E402
from svgplot import fmt  # noqa: E402

W, H = 1000, 350
CELL_W, CELL_H, STRIDE = 74, 38, 77
X0 = 150
ROWS = (("original", 60), ("selected?", 120), ("input", 180), ("target", 252))
FILL = {"mask": "#fefcbf", "random": "#fed7d7", "keep": "#c6f6d5", "": "#ffffff"}


def cell(d, x, y, fill, text, size=13, bold=False):
    d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{CELL_W}" height="{CELL_H}" fill="{fill}" stroke="{FRAME}" stroke-width="1"/>')
    d.text(x + CELL_W / 2, y + CELL_H / 2, [text], size=size, color=INK, weight="bold" if bold else "normal")


def render() -> str:
    tried, ids, x, m, kinds = encoders.corruption_example_data()
    n = len(ids)
    d = Diagram(W, H)
    ys = {name: y for name, y in ROWS}
    for name, y in ROWS:
        d.text(X0 - 12, y + CELL_H / 2, [name], size=13, color=MUTED, anchor="end")
    for i in range(n):
        px = X0 + i * STRIDE
        d.text(px + CELL_W / 2, 36, [f"position {i}"], size=11, color=MUTED)
        cell(d, px, ys["original"], "#ffffff", encoders.WP_A[ids[i]])
        d.text(px + CELL_W / 2, ys["selected?"] + CELL_H / 2, [kinds[i] if kinds[i] else "no"], size=13,
               color=INK if kinds[i] else MUTED, weight="bold" if kinds[i] else "normal")
        cell(d, px, ys["input"], FILL[kinds[i]], encoders.WP_A[x[i]], bold=bool(kinds[i]))
        cell(d, px, ys["target"], "#e2e8f0" if m[i] else "#ffffff", encoders.WP_A[ids[i]] if m[i] else "-")
        d.items.append(f'<line x1="{fmt(px + CELL_W / 2)}" y1="{fmt(ys["original"] + CELL_H)}" x2="{fmt(px + CELL_W / 2)}" '
                       f'y2="{fmt(ys["selected?"] - 8)}" stroke="{FRAME}" stroke-width="1" stroke-dasharray="3 3"/>')
        d.items.append(f'<line x1="{fmt(px + CELL_W / 2)}" y1="{fmt(ys["selected?"] + 12)}" x2="{fmt(px + CELL_W / 2)}" '
                       f'y2="{fmt(ys["input"])}" stroke="{FRAME}" stroke-width="1" stroke-dasharray="3 3"/>')
    ly = 330
    for k, (name, label) in enumerate((("mask", "[MASK]"), ("random", "random token"), ("keep", "left unchanged"))):
        lx = X0 + k * 190
        d.items.append(f'<rect x="{lx}" y="{ly - 12}" width="22" height="14" rx="3" fill="{FILL[name]}" stroke="{MUTED}" stroke-width="1.2"/>')
        d.text(lx + 30, ly - 5, [label], size=12, color=MUTED, anchor="start")
    d.items.append(f'<rect x="{X0 + 3 * 190}" y="{ly - 12}" width="22" height="14" rx="3" fill="#e2e8f0" stroke="{MUTED}" stroke-width="1.2"/>')
    d.text(X0 + 3 * 190 + 30, ly - 5, ["position with loss"], size=12, color=MUTED, anchor="start")
    return d.svg()
