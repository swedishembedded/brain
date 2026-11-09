# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Four steps of greedy generation drawn as a grid of positions, once recomputing the whole text at every step and once with a
# cache. A cell is a position of the text at one step: dark when its keys and values are computed in that step, light when they are
# read from the cache, empty when the token does not exist yet. The tokens, the rows computed per step and the totals come from
# `kv_cache.generate_naive` and `kv_cache.generate_cached`, the functions behind the chapter's tables.
#
# Swedish Embedded AB implements solutions for serving language models without recomputing the past. If your team needs expertise
# in key-value caches and decode performance, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import kv_cache as kc  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

W, H = 1000, 410
CELL_W, CELL_H, GAP = 52, 36, 6
PANEL_W, PANEL_H = 476, 290
PANEL_Y = 58
COMPUTED, REUSED = "#9ec5e8", "#eef2f7"


def cell(d, x, y, fill, label):
    d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{CELL_W}" height="{CELL_H}" rx="4" fill="{fill}" stroke="{FRAME}" stroke-width="1"/>')
    if label:
        d.items.append(f'<text x="{fmt(x + CELL_W / 2)}" y="{fmt(y + CELL_H / 2 + 4)}" {FONT} font-size="12" fill="{INK}" '
                       f'text-anchor="middle">{escape(label)}</text>')


def panel(d, x0, title, per_step, tokens, total):
    """per_step: for each step, (cached rows before the step, rows computed in it)."""
    d.panel(x0, PANEL_Y, PANEL_W, PANEL_H, title)
    n = len(tokens)
    left = x0 + 84
    d.text(left - 12, PANEL_Y + 52, ["position"], size=12, color=MUTED, anchor="end")
    for i, tok in enumerate(tokens):
        d.text(left + i * (CELL_W + GAP) + CELL_W / 2, PANEL_Y + 52, [f"{i}"], size=12, color=MUTED)
    for s, (reused, computed) in enumerate(per_step):
        y = PANEL_Y + 66 + s * (CELL_H + GAP)
        d.text(left - 12, y + CELL_H / 2, [f"step {s + 1}"], size=12, color=MUTED, anchor="end")
        for i in range(n):
            x = left + i * (CELL_W + GAP)
            if i < reused:
                cell(d, x, y, REUSED, "cache")
            elif i < reused + computed:
                cell(d, x, y, COMPUTED, str(tokens[i]))
    d.text(x0 + PANEL_W / 2, PANEL_Y + PANEL_H - 22, [f"rows computed in all: {total}"], size=13, weight="bold")


def render() -> str:
    p = kc.model()
    toks, steps_n = kc.generate_naive(p, kc.CFG, kc.PROMPT, kc.N_SHORT)
    tokens = list(kc.PROMPT) + toks[:kc.N_SHORT - 1]
    naive = [(0, n) for n, _, _ in steps_n]
    _, steps_c = kc.generate_cached(p, kc.CFG, kc.PROMPT, kc.N_SHORT)
    cached = [(clen - new, new) for new, clen, _, _ in steps_c]
    d = Diagram(W, H)
    d.text(W / 2, 24, [f"Keys and values computed at each step, prompt of {len(kc.PROMPT)} tokens and {kc.N_SHORT} generated"], size=14, weight="bold")
    panel(d, 16, "Recompute the whole text every step", naive, tokens, sum(c for _, c in naive))
    panel(d, 508, "Keep the keys and values: a cache", cached, tokens, sum(c for _, c in cached))
    d.swatch(W / 2 - 150, H - 36, 22, 14, "accent")
    d.items[-1] = d.items[-1].replace("#ebf4ff", COMPUTED)
    d.text(W / 2 - 118, H - 29, ["computed this step (token id shown)"], size=12, color=MUTED, anchor="start")
    d.swatch(W / 2 + 120, H - 36, 22, 14, "plain")
    d.items[-1] = d.items[-1].replace('fill="#ffffff"', f'fill="{REUSED}"')
    d.text(W / 2 + 150, H - 29, ["read from the cache"], size=12, color=MUTED, anchor="start")
    return d.svg()
