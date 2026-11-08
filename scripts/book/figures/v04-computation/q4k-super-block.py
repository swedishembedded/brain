# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The 144 bytes of one Q4_K super-block, byte by byte, in three zooms on one grid.

Row 1 is the whole block with the widths of its fields proportional to their bytes; row 2 zooms on the 16 header bytes
(f16 d, f16 dmin, twelve bytes of packed scales); row 3 zooms on the twelve scale bytes, three rows of four, each byte split
at the bit where its two fields meet; row 4 shows the first four of the 128 code bytes, each split into a low and a high
nibble. The field offsets and sizes are read from the worked module `quantization` (`Q4K_FIELDS`), so the picture and the
chapter's byte tables cannot disagree. All four columns of rows 3 and 4 are the same width.

Swedish Embedded AB implements solutions for compact, exactly specified weight formats in machine learning engines. If
your team needs expertise in quantized checkpoint formats, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import quantization as qz  # noqa: E402
from svgdiagram import FILL, INK, MUTED, Diagram, text_width  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

X0, X1 = 130, 910  # the content columns; the gutter at the left holds the row labels
W = X1 - X0


def _cell(d, x, y, w, h, lines, kind, size=12):
    """A rectangle with square-ish corners and centred text that must fit its width."""
    d.items.append(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="3" '
                   f'fill="{FILL[kind]}" stroke="{MUTED}" stroke-width="1.2"/>')
    assert max(text_width(ln, size) for ln in lines) <= w - 4, (lines, w)
    d.text(x + w / 2, y + h / 2, lines, size=size)


def render() -> str:
    off = qz.q4k_offsets()
    total = sum(size for _, size in qz.Q4K_FIELDS)
    hdr = off["qs"][0]  # the header is every byte before the codes
    d = Diagram(X1 + 14, 450)

    # row 1: the whole block, widths proportional to bytes
    y1, h1 = 24, 34
    px = W / total
    d.text(14, y1 + h1 / 2, ["whole block", f"{total} bytes"], anchor="start", size=12, weight="bold")
    _cell(d, X0, y1, hdr * px, h1, ["header", f"bytes 0..{hdr - 1}"], "accent")
    _cell(d, X0 + hdr * px, y1, (total - hdr) * px, h1,
          [f"codes: bytes {hdr}..{total - 1} (128 bytes, 256 nibbles)"], "model")

    # row 2: the header bytes
    y2, h2 = 100, 40
    cw = W / hdr
    d.line((X0, y1 + h1), (X0, y2))
    d.line((X0 + hdr * px, y1 + h1), (X1, y2))
    d.text(14, y2 + h2 / 2, ["header", f"bytes 0..{hdr - 1}"], anchor="start", size=12, weight="bold")
    names = []
    for name, size in qz.Q4K_FIELDS[:3]:
        if size == 2:
            names += [f"{name} lo", f"{name} hi"]
        else:
            names += [f"s{k}" for k in range(size)]
    for b, nm in enumerate(names):
        kind = "result" if b < 4 else "model"
        _cell(d, X0 + b * cw, y2, cw, h2, [nm, f"byte {b}"], kind, size=11)

    # row 3: the twelve scale bytes in three rows of four, columns of equal width
    s0 = off["scales"][0]
    y3, h3, gap3 = 188, 30, 6
    colw = W / 4
    d.line((X0 + s0 * cw, y2 + h2), (X0, y3))
    d.line((X1, y2 + h2), (X1, y3))
    rows3 = [
        ("sc_j, high bits of sc_(j+4)", [(6, f"sc{j}", "accent", 12) for j in range(4)], [(2, f"sc{j + 4} hi", "accent", 11) for j in range(4)]),
        ("m_j, high bits of m_(j+4)", [(6, f"m{j}", "plain", 12) for j in range(4)], [(2, f"m{j + 4} hi", "plain", 11) for j in range(4)]),
        ("low nibbles of sc_(j+4), m_(j+4)", [(4, f"sc{j + 4} lo", "accent", 11) for j in range(4)], [(4, f"m{j + 4} lo", "plain", 11) for j in range(4)]),
    ]
    for r, (label, lows, highs) in enumerate(rows3):
        y = y3 + r * (h3 + gap3)
        first = s0 + 4 * r
        d.text(14, y + h3 / 2, [f"bytes {first}..{first + 3}"], anchor="start", size=12, weight="bold")
        for c in range(4):
            x = X0 + c * colw
            wl, nl, kl, sl = lows[c]
            wr, nr, kr, sr = highs[c]
            # the first field of a byte sits in its low bits, the second in its high bits: high bits are drawn on the left
            _cell(d, x, y, colw * wr / 8, h3, [nr], kr, size=sr)
            _cell(d, x + colw * wr / 8, y, colw * wl / 8, h3, [nl], kl, size=sl)
    d.text(X0, y3 + 3 * (h3 + gap3) + 2, ["every byte is drawn with its high bits on the left: bits 7..6 | 5..0 in the first two rows, bits 7..4 | 3..0 in the third row and in the code bytes below"],
           anchor="start", size=11, color=MUTED)

    # row 4: the first four code bytes, split into a high and a low nibble
    y4, h4 = 330, 30
    d.text(14, y4 + h4 / 2, [f"bytes {hdr}..{hdr + 3}", "first of the codes"], anchor="start", size=12, weight="bold")
    for c in range(4):
        x = X0 + c * colw
        _cell(d, x, y4, colw / 2, h4, [f"w{32 + c} (high)"], "accent", size=11)
        _cell(d, x + colw / 2, y4, colw / 2, h4, [f"w{c} (low)"], "model", size=11)
    notes = [
        f"bytes {hdr}..{hdr + 31}: low nibbles are weights 0..31 (sub-block 0), high nibbles weights 32..63 (sub-block 1);",
        f"bytes {hdr + 32}..{hdr + 127}: the same pattern for weights 64..127, 128..191 and 192..255 (sub-blocks 2 to 7);",
        "weight i of sub-block j = i // 32 decodes to d sc_j q - dmin m_j.",
    ]
    for i, t in enumerate(notes):
        d.text(X0, y4 + h4 + 24 + i * 18, [t], anchor="start", size=12, color=INK)
    return d.svg()
