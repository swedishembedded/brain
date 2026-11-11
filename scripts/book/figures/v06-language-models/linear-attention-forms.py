# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two forms of linear attention, which compute the same outputs: the parallel form builds the whole T x T score matrix, masks
# it and multiplies by the values; the recurrent form carries a d_k x d_v state from token to token. The multiply-add counts in the
# captions are those of `linear_attention.two_forms_macs` for the shape of the chapter's random check (T = 64, d_k = d_v = 8).
#
# Swedish Embedded AB implements solutions for serving long-context language models within a memory budget. If your team needs
# expertise in linear attention, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import linear_attention as la  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, PANEL_H, GAP, TOP = 780, 176, 18, 16
COLS = (97, 242, 387, 532, 677)
BOX_W, BOX_H = 120, 62


def render() -> str:
    T, dk, dv = 64, 8, 8
    parallel, recurrent = la.two_forms_macs(T, dk, dv)
    d = Diagram(W, TOP + 2 * PANEL_H + GAP + 16)

    y = TOP
    d.panel(20, y, W - 40, PANEL_H, "Parallel form: all tokens at once")
    cy = y + 82
    qk = d.box(COLS[0], cy, BOX_W, BOX_H, ["Q, K", f"[T, d_k]"], "accent")
    sc = d.box(COLS[1], cy, BOX_W, BOX_H, ["scores Q K^T", "[T, T]"], "model")
    mk = d.box(COLS[2], cy, BOX_W, BOX_H, ["causal mask", "no softmax"], "model")
    pv = d.box(COLS[3], cy, BOX_W, BOX_H, ["times V", "[T, d_v]"], "model")
    out = d.box(COLS[4], cy, BOX_W, BOX_H, ["O", "[T, d_v]"], "result")
    d.arrow(qk["r"], sc["l"])
    d.arrow(sc["r"], mk["l"])
    d.arrow(mk["r"], pv["l"])
    d.arrow(pv["r"], out["l"])
    d.text(W / 2, y + PANEL_H - 24, [f"T = {T}, d_k = d_v = {dk}: {parallel:,} multiply-adds; the T x T matrix grows with the square of T"],
           size=12, color=MUTED)

    y = TOP + PANEL_H + GAP
    d.panel(20, y, W - 40, PANEL_H, "Recurrent form: one token at a time")
    cy = y + 82
    prev = d.oval(COLS[0], cy, BOX_W, BOX_H, ["S(t-1)", "[d_k, d_v]"], "accent")
    wr = d.box(COLS[1], cy, BOX_W, BOX_H, ["write", "+ k(t) v(t)^T"], "model")
    cur = d.oval(COLS[2], cy, BOX_W, BOX_H, ["S(t)", "[d_k, d_v]"], "accent")
    rd = d.box(COLS[3], cy, BOX_W, BOX_H, ["read", "S(t)^T q(t)"], "model")
    ot = d.oval(COLS[4], cy, BOX_W, BOX_H, ["o(t)", "[d_v]"], "result")
    d.arrow(prev["r"], wr["l"])
    d.arrow(wr["r"], cur["l"])
    d.arrow(cur["r"], rd["l"])
    d.arrow(rd["r"], ot["l"])
    d.path([(cur["t"][0], cur["t"][1]), (cur["t"][0], y + 46), (prev["t"][0], y + 46), (prev["t"][0], prev["t"][1])], dash="5 3")
    d.text((prev["t"][0] + cur["t"][0]) / 2, y + 38, ["S(t) becomes S(t-1) of the next token"], size=12, color=MUTED)
    d.text(W / 2, y + PANEL_H - 24, [f"same T and widths: {recurrent:,} multiply-adds; the state does not grow with T"], size=12, color=MUTED)
    return d.svg()
