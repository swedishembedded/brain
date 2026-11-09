# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Multi-head latent attention in two settings. Left: what is written for each token and how the keys and values are rebuilt from
# the cached latent. Right: the decode path in which the key up-projection is absorbed into the query and the value up-projection
# into the output, so that attention reads only the latent and the rotary key. Yellow ovals are what the cache holds. Widths are
# those of the tiny layer of the chapter's worked example (`attention_variants`: d_c = 5, d_R = 2).
#
# Swedish Embedded AB implements solutions for serving language models within a memory budget. If your team needs expertise in
# key-value cache design, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import attention_variants as av  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 960, 640
PW, PH = 460, 620
LX, RX = 10, 490


def render() -> str:
    dc, dr = av.DC, av.DR
    d = Diagram(W, H)
    d.panel(LX, 10, PW, PH, "Rebuild every key and value (training, prefill)")
    d.panel(RX, 10, PW, PH, "Absorb the up-projections (decode)")

    # left panel: write path, then read path
    cx = LX + PW / 2
    x = d.oval(cx, 85, 150, 38, ["x_j  (token j)"], "accent")
    down = d.box(cx - 105, 165, 150, 44, ["W_DKV", f"d_c = {dc}"])
    rope = d.box(cx + 105, 165, 150, 44, ["W_KR, then RoPE", f"d_R = {dr}"])
    c = d.oval(cx - 105, 250, 150, 40, ["c_j  (cached)"], "result")
    kr = d.oval(cx + 105, 250, 150, 40, ["k_R,j  (cached)"], "result")
    uk = d.box(cx - 165, 335, 100, 40, ["W_UK"])
    uv = d.box(cx - 45, 335, 100, 40, ["W_UV"])
    k = d.oval(cx - 165, 420, 100, 38, ["k_j"])
    v = d.oval(cx - 45, 420, 100, 38, ["v_j"])
    att = d.box(cx, 505, 400, 50, ["q . k_j + q_R . k_R,j, softmax,", "weighted sum of the v_j  (all H heads)"], "model")
    wo = d.box(cx, 575, 150, 40, ["W_O"])
    d.arrow(x["b"], down["t"])
    d.arrow(x["b"], rope["t"])
    d.arrow(down["b"], c["t"])
    d.arrow(rope["b"], kr["t"])
    d.arrow(c["b"], uk["t"])
    d.arrow(c["b"], uv["t"])
    d.arrow(uk["b"], k["t"])
    d.arrow(uv["b"], v["t"])
    d.arrow(k["b"], (k["b"][0], att["t"][1]))
    d.arrow(v["b"], (v["b"][0], att["t"][1]))
    d.arrow(kr["b"], (kr["b"][0], att["t"][1]))
    d.arrow(att["b"], wo["t"])

    # right panel: the query is carried into the latent space
    cx = RX + PW / 2
    ax, bx, cxr = cx - 135, cx, cx + 135
    q = d.oval(ax, 85, 120, 44, ["q, q_R", "(new token)"], "accent")
    cj = d.oval(bx, 85, 120, 38, ["c_j  (cached)"], "result")
    kj = d.oval(cxr, 85, 120, 38, ["k_R,j  (cached)"], "result")
    wuk = d.box(ax, 165, 120, 44, ["W_UK^T", "on q only"])
    qt = d.oval(ax, 250, 120, 38, [f"q~  (d_c = {dc})"])
    sc = d.box(cx, 335, 420, 50, ["q~ . c_j + q_R . k_R,j, softmax", "(one latent shared by all H heads)"], "model")
    avg = d.box(cx, 420, 300, 44, ["weighted sum of the c_j", f"(d_c = {dc} per head)"], "model")
    out = d.box(cx, 505, 300, 50, ["W_UV on each head, then W_O,", "or W_OV = W_O W_UV in one product"], "model")
    res = d.oval(cx, 580, 150, 38, ["output  (D)"])
    d.arrow(q["b"], wuk["t"])
    d.arrow(wuk["b"], qt["t"])
    d.arrow(qt["b"], (qt["b"][0], sc["t"][1]))
    d.arrow(cj["b"], (cj["b"][0], sc["t"][1]))
    d.arrow(kj["b"], (kj["b"][0], sc["t"][1]))
    d.arrow(sc["b"], avg["t"])
    d.arrow(avg["b"], out["t"])
    d.arrow(out["b"], res["t"])
    d.text(cx, 622, ["no key or value is ever rebuilt for a cached token"], size=12, color=MUTED)
    return d.svg()
