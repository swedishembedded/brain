# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The gradient of the lookup `cat sat cat .`: the gradient row of each position is added into the table row of its
# id. `cat` is read twice, so its row receives the sum of two; unused rows stay zero. Numbers from
# `embeddings.gradient_data`, the chapter's own scatter-add.
#
# Swedish Embedded AB implements solutions for training token tables without lost gradients. If your team needs
# expertise in embedding backward passes, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import embeddings  # noqa: E402
from svgdiagram import Diagram, RED  # noqa: E402

W, H = 780, 318
ROW_H = 38


def _v(row):
    return "(" + ", ".join(f"{x:.2f}".replace("-0.00", "0.00") for x in row) + ")"


def render() -> str:
    g = embeddings.gradient_data()
    ids, dX, dE, toks = g["ids"], g["dX"], g["dE"], g["tokens"]
    d = Diagram(W, H)
    top = 62
    lx, lw = 150, 280
    rx, rw = 620, 280
    d.text(lx, 26, ["gradient arriving at each position"], size=12, color="#4a5568")
    d.text(rx, 26, ["gradient of the table"], size=12, color="#4a5568")
    left = []
    for t, i in enumerate(ids):
        y = top + t * (ROW_H + 18) + 12
        left.append(d.box(lx, y, lw, 30, [f"t = {t}, id {i}:  dX = {_v(dX[t])}"], "accent"))
    right = []
    for i in range(len(toks)):
        y = top + i * 36 - 6
        used = i in ids
        right.append(d.box(rx, y, rw, 28, [f"{i} {toks[i]:<3}  dW = {_v(dE[i])}"], "result" if used else "plain"))
    for t, i in enumerate(ids):
        d.arrow(left[t]["r"], (right[i]["l"][0], right[i]["l"][1] + (-6 if ids.count(i) > 1 and ids.index(i) == t else 6 if ids.count(i) > 1 else 0)),
                color=RED, gap=2)
    d.text(W / 2 + 14, H - 14, ["row 1 is read twice, so it receives dX[0] + dX[2]; rows 0, 2 and 4 are never read and stay zero"],
           size=12, color="#2d3748")
    return d.svg()
