# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Where the parameters of four models sit, as shares of each model's own total: the stated small decoder, Qwen3 8B, the text tower
# of Gemma-4 and the diffusion transformer of MiniMax-H3. Numbers from `model_cost.params_share_data`, the function behind the
# tables of the chapter. The sliver of "norms and other" is the aggregate projection for Gemma-4 and the input, output and time
# layers for MiniMax-H3; the hatched-looking fifth colour is the `adaln_proj` that MiniMax-H3 never holds.
#
# Swedish Embedded AB implements solutions for sizing language models before they are run. If your team needs expertise in
# parameter and memory budgeting, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import model_cost as mc  # noqa: E402
from svgdiagram import MUTED, INK, Diagram  # noqa: E402
from svgplot import FONT  # noqa: E402

W, H = 780, 360
LABEL_X, BAR_X, BAR_W = 20, 250, 500
ROWS = (95, 155, 215, 275)
BAR_H = 34
FILLS = {"embedding and head": "#bfd7ea", "attention matrices": "#fdd0a2", "feed-forward matrices": "#c7e9c0",
         "norms and other": "#e2e8f0", "adaln_proj": "#d6c9ee"}


def _human(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f} billion"
    if n >= 1e6:
        return f"{n / 1e6:.2f} million"
    return f"{n:,}"


def render() -> str:
    data = mc.params_share_data()
    d = Diagram(W, H)
    d.text(W / 2, 26, ["Share of each model's own parameters, by component"], size=14, weight="bold")
    for y, (label, total, parts) in zip(ROWS, data):
        d.text(LABEL_X, y - 7, [label], size=13, weight="bold", anchor="start")
        d.text(LABEL_X, y + 11, [_human(total)], size=12, color=MUTED, anchor="start")
        x = BAR_X
        for name, share in parts:
            w = BAR_W * share
            if w <= 0:
                continue
            d.items.append(f'<rect x="{x:.2f}" y="{y - BAR_H / 2}" width="{w:.2f}" height="{BAR_H}" fill="{FILLS[name]}" '
                           f'stroke="{MUTED}" stroke-width="1"/>')
            if share >= 0.06:
                d.text(x + w / 2, y, [f"{100 * share:.0f}%"], size=12)
            x += w
    # legend, two rows of entries
    lx, ly = BAR_X, 312
    for i, (name, c) in enumerate(FILLS.items()):
        cx = BAR_X + (i % 3) * 170 if i < 3 else BAR_X + (i - 3) * 170
        cy = 322 if i < 3 else 342
        d.items.append(f'<rect x="{cx}" y="{cy - 8}" width="14" height="14" rx="2" fill="{c}" stroke="{MUTED}" stroke-width="1"/>')
        d.items.append(f'<text x="{cx + 20}" y="{cy + 4}" {FONT} font-size="12" fill="{INK}">{name}</text>')
    return d.svg()
