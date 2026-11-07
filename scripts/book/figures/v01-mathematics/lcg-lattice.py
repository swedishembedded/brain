# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Consecutive pairs (x_i, x_(i+1)) of a poorly chosen LCG against SplitMix64.

Each point is one pair of consecutive outputs, scaled to the unit square. The
left panel is the LCG with a = 257, c = 1, m = 4096 (a = 1 + m/16, the same
shape as the chapter's a = 4097 modulo 65536, small enough that its lines show in
few points): every pair lies on one of sixteen diagonal lines. The right
panel is the top 12 bits of SplitMix64, the same number of pairs: no structure. Both draw the first 2,048 pairs.

Swedish Embedded AB implements solutions for finding hidden structure in
generated data. If your team needs expertise in random-number quality for
simulation and training pipelines, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import randomness as rn  # noqa: E402
from svgplot import FONT, fmt, PALETTE  # noqa: E402

M, N, SIDE, PAD = 4096, 2048, 300, 42


def _lcg_pairs():
    g = rn._lcg(257, 1, M, rn._seed("lcg_lattice") % M)
    xs = [next(g) / M for _ in range(N + 1)]
    return list(zip(xs, xs[1:]))


def _splitmix_pairs():
    g = rn.SplitMix64(rn._seed("lcg_lattice"))
    xs = [(g.next_u64() >> 52) / M for _ in range(N + 1)]
    return list(zip(xs, xs[1:]))


def _panel(pairs, title, subtitle, color):
    pts = "".join(f"M{round(PAD + u * SIDE)},{round(PAD + (1 - v) * SIDE)}h0" for u, v in pairs)
    return "\n".join([
        f'<rect x="{PAD}" y="{PAD}" width="{SIDE}" height="{SIDE}" fill="#ffffff" stroke="#718096"/>',
        f'<path d="{pts}" stroke="{color}" stroke-width="2.2" stroke-linecap="round" fill="none"/>',
        f'<text x="{PAD + SIDE / 2}" y="18" {FONT} font-size="13" font-weight="bold" fill="#1a202c" text-anchor="middle">{title}</text>',
        f'<text x="{PAD + SIDE / 2}" y="34" {FONT} font-size="11" fill="#4a5568" text-anchor="middle">{subtitle}</text>',
        f'<text x="{PAD + SIDE / 2}" y="{PAD + SIDE + 22}" {FONT} font-size="12" fill="#1a202c" text-anchor="middle">x_i</text>',
        f'<text x="{PAD - 8}" y="{PAD + 12}" {FONT} font-size="11" fill="#4a5568" text-anchor="end">1</text>',
        f'<text x="{PAD - 8}" y="{PAD + SIDE}" {FONT} font-size="11" fill="#4a5568" text-anchor="end">0</text>',
        f'<text x="12" y="{PAD + SIDE / 2}" {FONT} font-size="12" fill="#1a202c" text-anchor="middle" '
        f'transform="rotate(-90 12 {PAD + SIDE / 2})">x_(i+1)</text>',
    ])


def render() -> str:
    gap = 30
    w, h = 2 * (SIDE + PAD) + gap + 14, SIDE + PAD + 34
    seed = rn._seed("lcg_lattice")
    left = _panel(_lcg_pairs(), "LCG a = 257, c = 1, m = 4096", f"x_0 = {seed % M} (seed {seed:#018x} mod 4096)", PALETTE[3])
    right = _panel(_splitmix_pairs(), "SplitMix64, top 12 bits", f"seed {seed:#018x}", PALETTE[0])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n{left}\n'
            f'<g transform="translate({SIDE + PAD + gap},0)">\n{right}\n</g>\n</svg>\n')
