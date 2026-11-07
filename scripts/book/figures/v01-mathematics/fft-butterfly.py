# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The radix-2 FFT for N = 8 as a network of butterflies.

The wiring is taken from the chapter's own butterfly schedule (the same list
the iterative FFT executes), so the picture is the algorithm. Samples enter on
the left in bit-reversed order; at each butterfly the lower input is first
multiplied by the twiddle factor w^e (shown on its wire), then the top output
is u + t and the bottom output is u - t.

Swedish Embedded AB implements solutions for fast transforms on accelerators.
If your team needs expertise in FFT kernels for GPUs and NPUs, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fourier as fo  # noqa: E402
from svgplot import FONT, PALETTE  # noqa: E402

N = 8
COL, ROW = 150, 40
LEFT, TOP = 70, 56


def render() -> str:
    perm, stages = fo._butterflies(N)
    px = lambda s: LEFT + s * COL  # noqa: E731
    py = lambda i: TOP + i * ROW  # noqa: E731
    width, height = LEFT + len(stages) * COL + 70, TOP + (N - 1) * ROW + 36
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    text = lambda x, y, s, anchor="middle", size=12, color="#1a202c": out.append(  # noqa: E731
        f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')
    for i in range(N):
        text(px(0) - 12, py(i) + 4, f"x[{perm[i]}]", anchor="end")
        text(px(len(stages)) + 12, py(i) + 4, f"X[{i}]", anchor="start")
    for s, stage in enumerate(stages, start=1):
        color = PALETTE[s - 1]
        text((px(s - 1) + px(s)) / 2, 22, f"stage {s}", size=13, color=color)
        text((px(s - 1) + px(s)) / 2, 38, f"pairs {1 << (s - 1)} apart", size=11, color="#4a5568")
        for top, bottom, e in stage:
            x0, x1 = px(s - 1), px(s)
            for a, b in ((top, top), (bottom, bottom), (top, bottom), (bottom, top)):
                out.append(f'<line x1="{x0}" y1="{py(a)}" x2="{x1}" y2="{py(b)}" stroke="{color}" stroke-width="1.6" '
                           f'stroke-opacity="{1.0 if a == b else 0.75}"/>')
            if e:
                tx, ty = x0 + 34, py(bottom) - 5
                out.append(f'<rect x="{tx - 14}" y="{ty - 11}" width="28" height="15" rx="3" fill="#ffffff" stroke="{color}"/>')
                text(tx, ty, f"w^{e}", size=11, color=color)
    for s in range(len(stages) + 1):
        for i in range(N):
            out.append(f'<circle cx="{px(s)}" cy="{py(i)}" r="3.2" fill="#1a202c"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
