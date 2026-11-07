# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Two distributions on four outcomes and the two directions of KL divergence.

P is sharply peaked and Q is uniform; the bar heights and both divergences are
computed with the chapter's own functions, so the picture matches the table.

Swedish Embedded AB implements solutions for comparing a model's beliefs with
the data. If your team needs expertise in divergence-based evaluation, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import information as inf  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402


YMAX = 1.15


def _text(x, y, s, size=12, anchor="middle", color="#1a202c", weight="normal"):
    return (f'<text x="{fmt(x)}" y="{fmt(y)}" {FONT} font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')


def render() -> str:
    p, q = inf.KL_P, inf.KL_Q
    d_pq, d_qp = inf.kl_divergence(p, q), inf.kl_divergence(q, p)
    w, h = 540, 330
    ox, oy, plot_w, plot_h = 56, 40, 440, 190   # plot area; y runs 0..1
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
           _text(w / 2, 22, "Same two distributions, two different divergences", 13, weight="bold")]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = oy + plot_h * (1 - t / YMAX)
        out.append(f'<line x1="{ox}" y1="{fmt(y)}" x2="{ox + plot_w}" y2="{fmt(y)}" stroke="#edf2f7"/>')
        out.append(_text(ox - 6, y + 4, f"{t:g}", 11, "end", "#4a5568"))
    slot = plot_w / len(p)
    bar = 44
    for i, (pi, qi) in enumerate(zip(p, q)):
        cx = ox + slot * (i + 0.5)
        for dx, v, col in ((-bar - 2, pi, "#2b6cb0"), (2, qi, "#c05621")):
            bh = plot_h * v / YMAX
            out.append(f'<rect x="{fmt(cx + dx)}" y="{fmt(oy + plot_h - bh)}" width="{bar}" height="{fmt(bh)}" fill="{col}"/>')
            out.append(_text(cx + dx + bar / 2, oy + plot_h - bh - 4, f"{v:g}", 11))
        out.append(_text(cx, oy + plot_h + 16, f"outcome {i + 1}", 11, color="#4a5568"))
    out.append(f'<rect x="{ox}" y="{oy}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#718096"/>')
    # legend
    out.append(f'<rect x="{ox + 250}" y="{oy + 6}" width="12" height="12" fill="#2b6cb0"/>')
    out.append(_text(ox + 268, oy + 16, "P (peaked)", 12, "start"))
    out.append(f'<rect x="{ox + 250}" y="{oy + 26}" width="12" height="12" fill="#c05621"/>')
    out.append(_text(ox + 268, oy + 36, "Q (uniform)", 12, "start"))
    out.append(_text(ox + plot_w / 2, h - 52, f"KL(P || Q) = {d_pq:.3f} bits   (data from P, code built for Q)", 13))
    out.append(_text(ox + plot_w / 2, h - 28, f"KL(Q || P) = {d_qp:.3f} bits   (data from Q, code built for P)", 13))
    out.append("</svg>")
    return "\n".join(out) + "\n"
