# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The data flow of one attention head on the four-token example: three learned projections of the same input give queries,
# keys and values; queries against keys give a score for every pair, a softmax turns each row into weights, and the weights
# average the values. Shapes are those of the chapter's example (T = 4, D = 3, d_k = d_v = 2).
#
# Swedish Embedded AB implements solutions for models built on attention, from the kernels to the training loop. If your
# team needs expertise in transformer attention, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import self_attention as sa  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 400
YQ, YK, YV = 90, 200, 310


def render() -> str:
    e = sa.example()
    T, D, DK, DV = len(e["X"]), len(e["X"][0]), len(e["Q"][0]), len(e["V"][0])
    d = Diagram(W, H)
    x = d.oval(62, YK, 90, 64, ["X", f"{T} x {D}"], "accent")
    proj, vals = {}, {}
    for name, y, dim in (("Q", YQ, DK), ("K", YK, DK), ("V", YV, DV)):
        proj[name] = d.box(190, y, 100, 46, [f"times W_{name}"], "model")
        vals[name] = d.oval(335, y, 96, 52, [name, f"{T} x {dim}"], "accent")
        d.arrow(x["r"], proj[name]["l"])
        d.arrow(proj[name]["r"], vals[name]["l"])
    # scores: a taller box so that Q and K arrive at their own points of its left edge
    sc_x, row = 505, 145
    scores = d.box(sc_x, row, 120, 120, ["Q K^T", "divided by", f"sqrt(d_k) = sqrt({DK})"], "model")
    d.arrow(vals["Q"]["r"], (sc_x - 60, row - 30))
    d.arrow(vals["K"]["r"], (sc_x - 60, row + 30))
    soft = d.box(660, row, 100, 60, ["softmax", "of each row"], "model")
    mask = d.oval(660, 45, 100, 36, ["mask (optional)"], "plain")
    d.arrow(mask["b"], soft["t"], "adds -inf where hidden", side="right")
    d.arrow(scores["r"], soft["l"], f"S  {T} x {T}", side="above")
    avg = d.box(800, row, 100, 60, ["P V", "weighted sums"], "model")
    d.arrow(soft["r"], avg["l"], f"P  {T} x {T}", side="above")
    out = d.oval(935, row, 90, 52, ["O", f"{T} x {DV}"], "result")
    d.arrow(avg["r"], out["l"])
    vx, vy = vals["V"]["r"]
    d.path([(vx, vy), (800, vy), (800, row + 30)])
    d.text(W / 2, 382, ["every row of Q asks, every row of K advertises, every row of V is what gets handed over; "
                        "the same X feeds all three"], size=12, color=MUTED)
    return d.svg()
