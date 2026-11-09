# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Memory by component for five cases: three inference cases (the text tower of Gemma-4 at bf16 and at int8 with a 32,768-token cache
# it would need if it decoded; the MiniMax-H3 backbone at int8 over 13,312 packed rows) and two training cases of the Gemma-4 tower
# (mixed precision, 8,192 tokens, activations kept or recomputed). Numbers from `model_cost.memory_breakdown_data`, the function behind
# the tables of the chapter; bar length is proportional to GiB.
#
# Swedish Embedded AB implements solutions for fitting large models into a fixed memory budget. If your team needs expertise in
# inference and training memory planning, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import model_cost as mc  # noqa: E402
from svgdiagram import MUTED, INK, Diagram  # noqa: E402
from svgplot import FONT  # noqa: E402

W, H = 800, 420
LABEL_X, BAR_X, BAR_MAX = 20, 20, 560
ROWS = (112, 180, 248, 316, 384)
BAR_H = 28
FILLS = {"weights": "#bfd7ea", "KV cache": "#fdd0a2", "workspace": "#c7e9c0", "gradients": "#fcbba1",
         "optimizer state and master": "#d6c9ee", "activations": "#c7e9c0"}
LEGEND = (("weights", "#bfd7ea"), ("KV cache", "#fdd0a2"), ("workspace or activations", "#c7e9c0"), ("gradients", "#fcbba1"),
          ("optimizer state and master copy", "#d6c9ee"))
LEGEND_X = (20, 105, 190, 365, 460)


def render() -> str:
    data = mc.memory_breakdown_data()
    top = max(sum(v for _, v in parts) for _, parts in data)
    d = Diagram(W, H)
    d.text(W / 2, 24, ["Memory by component, GiB (bar length proportional to the total)"], size=14, weight="bold")
    for y, (label, parts) in zip(ROWS, data):
        total = sum(v for _, v in parts)
        d.text(LABEL_X, y - 28, [label], size=12, weight="bold", anchor="start")
        x = BAR_X
        for name, v in parts:
            w = BAR_MAX * v / top
            if w <= 0.5:
                continue
            d.items.append(f'<rect x="{x:.2f}" y="{y - BAR_H / 2 + 2}" width="{w:.2f}" height="{BAR_H}" fill="{FILLS[name]}" '
                           f'stroke="{MUTED}" stroke-width="1"/>')
            if w >= 60:
                d.text(x + w / 2, y + 2, [f"{v:.1f}"], size=12)
            x += w
        note = "" if len(parts) > 2 and parts[0][0] != "weights" or total > 100 else "  (" + ", ".join(f"{n} {v:.1f}" for n, v in parts if v > 0) + ")"
        d.text(x + 10, y + 2, [f"{total:.1f} GiB{note}"], size=13, anchor="start")
    for (name, c), cx in zip(LEGEND, LEGEND_X):
        d.items.append(f'<rect x="{cx}" y="{ROWS[0] - 62}" width="12" height="12" rx="2" fill="{c}" stroke="{MUTED}" stroke-width="1"/>')
        d.items.append(f'<text x="{cx + 17}" y="{ROWS[0] - 52}" {FONT} font-size="12" fill="{INK}">{name}</text>')
    return d.svg()
