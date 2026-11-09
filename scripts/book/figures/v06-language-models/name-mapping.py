# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The tensors of one layer of the published layout on the left and the tensors of the book's decoder on the right, with an arrow
# for every step of the import plan. The fused matrices fan out into their parts. Taken from `importing_models.mapping_data`, the
# function behind the table of the name mapping.
#
# Swedish Embedded AB implements solutions for importing third-party model checkpoints without silent drift. If your team needs
# expertise in weight-name mappings and layout conversions, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importing_models as im  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W = 800
LX, RX, BW_L, BW_R = 195, 640, 350, 280
TOP, STEP, BH = 80, 50, 40


def short(name):
    return name.replace("model.layers.0.", "layers.N.")


def render() -> str:
    steps = im.mapping_data()
    rows = {s[2]: i for i, s in enumerate(steps)}
    H = TOP + (len(steps) - 1) * STEP + 100
    d = Diagram(W, H)
    d.text(LX, 28, ["published layout (one layer)"], size=14, weight="bold")
    d.text(RX, 28, ["the book's decoder (layer 0)"], size=14, weight="bold")
    right = {}
    for i, (src, sshape, dest, dshape, op) in enumerate(steps):
        kind = "plain" if op == "copy" else ("result" if "rows" in op else "accent")
        right[dest] = d.box(RX, TOP + i * STEP, BW_R, BH, [dest.replace(" (layer 0)", ""), f"{' x '.join(map(str, dshape))}: {op}"], kind)
    # one left box per source tensor, centred on the mean row of its destinations
    order, members = [], {}
    for i, s in enumerate(steps):
        if s[0] not in members:
            order.append(s[0])
        members.setdefault(s[0], []).append(i)
    left = {}
    for src in order:
        idx = members[src]
        y = TOP + sum(idx) / len(idx) * STEP
        shape = next(s[1] for s in steps if s[0] == src)
        left[src] = d.box(LX, y, BW_L, BH, [short(src), "stored " + " x ".join(map(str, shape))], "model")
    for src, dest in ((s[0], s[2]) for s in steps):
        d.arrow(left[src]["r"], right[dest]["l"], gap=3)
    ky = H - 36
    for k, (kind, label) in enumerate((("plain", "copied"), ("accent", "transposed"), ("result", "rows cut out, transposed"))):
        x = 60 + k * 230
        d.swatch(x, ky - 9, 26, 18, kind)
        d.text(x + 34, ky, [label], size=12, color=MUTED, anchor="start")
    return d.svg()
