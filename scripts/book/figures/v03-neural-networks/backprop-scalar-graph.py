# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The scalar chain y = w2 tanh(w1 x) as a computational graph with forward values and backward gradients.

Every number is read from the worked-example module, which does the arithmetic.

Swedish Embedded AB implements solutions for explaining and verifying
gradient computation in neural network training. If your team needs
expertise in backpropagation, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import backprop  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402


def render() -> str:
    v = backprop.scalar_chain_values()
    # (op label, value name, forward value, gradient of L with respect to it)
    nodes = [("u = w1 x", v["u"], v["d_u"]), ("h = tanh(u)", v["h"], v["d_h"]), ("y = w2 h", v["y"], v["d_y"]),
             ("e = y - t", v["e"], v["d_e"]), ("L = e^2", v["L"], 1.0)]
    W, Hh = 780, 215
    xs = [90 + i * 150 for i in range(5)]
    y0 = 120
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}">',
           '<defs><marker id="af" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#1a202c"/></marker>'
           '<marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#c53030"/></marker></defs>',
           f'<rect width="{W}" height="{Hh}" fill="#ffffff"/>']

    def text(x, y, s, size=12, color="#1a202c", anchor="middle", weight="normal"):
        out.append(f'<text x="{fmt(x)}" y="{fmt(y)}" {FONT} font-size="{size}" fill="{color}" '
                   f'text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')

    # inputs entering each node from above: (node index, label)
    feeds = {0: [f"w1 = {v['w1']}", f"x = {v['x']}"], 2: [f"w2 = {v['w2']}"], 3: [f"t = {v['t']}"]}
    for i, (name, val, grad) in enumerate(nodes):
        x = xs[i]
        out.append(f'<rect x="{x - 58}" y="{y0 - 22}" width="116" height="44" rx="9" fill="#e2e8f0" stroke="#4a5568"/>')
        text(x, y0 - 3, name, 13)
        text(x, y0 + 14, f"= {val:.4f}", 12, "#1a202c")
        text(x, y0 + 52, f"dL/d{name.split()[0]} = {grad:.4f}", 12, "#c53030", weight="bold")
        for k, lab in enumerate(feeds.get(i, [])):
            fx = x - 40 + k * 80 if len(feeds[i]) > 1 else x
            out.append(f'<line x1="{fx}" y1="{y0 - 62}" x2="{x - 20 + k * 40 if len(feeds[i]) > 1 else x}" '
                       f'y2="{y0 - 23}" stroke="#1a202c" marker-end="url(#af)"/>')
            text(fx, y0 - 68, lab, 11, "#4a5568")
        if i < 4:
            out.append(f'<line x1="{x + 59}" y1="{y0 - 6}" x2="{xs[i + 1] - 60}" y2="{y0 - 6}" stroke="#1a202c" '
                       'marker-end="url(#af)"/>')
            out.append(f'<line x1="{xs[i + 1] - 60}" y1="{y0 + 40}" x2="{x + 59}" y2="{y0 + 40}" stroke="#c53030" '
                       'stroke-dasharray="5 3" marker-end="url(#ab)"/>')
    # parameter gradients
    text(xs[0] - 58, y0 + 80, f"dL/dw1 = dL/du * x = {v['d_w1']:.4f}", 12, "#c53030", anchor="start")
    text(xs[2], y0 + 80, f"dL/dw2 = dL/dy * h = {v['d_w2']:.4f}", 12, "#c53030")
    text(W / 2, 20, "Forward values (black, left to right) and gradients of L (red, right to left)", 13, weight="bold")
    out.append("</svg>")
    return "\n".join(out) + "\n"
