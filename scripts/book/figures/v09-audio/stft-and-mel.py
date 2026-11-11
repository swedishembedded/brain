# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One chirp (100 Hz to 3500 Hz in 0.4 s, 8000 Hz audio) as a short-time transform with N = 128 and hop 80, shown twice: left the
# power in each of the 65 bins, right the energy in each of 24 Slaney mel filters of the same frames. Darker is stronger, in decibels
# below the largest cell. The data are `spectrograms.figure_spectrograms`.
#
# Swedish Embedded AB implements solutions for turning raw audio into the pictures speech models read. If your team needs expertise in
# audio front ends for speech and music models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spectrograms as sp  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402

CELL_W = 6.0
PANEL_H = 168.0
LEVELS = 9  # decibel bands of 80 / 9 below the maximum, white to dark blue
GX1, GX2, GY = 64, 372, 52


def _shade(db):
    """White at -80 dB, dark blue at 0 dB, in LEVELS steps."""
    k = min(LEVELS - 1, int((db + 80.0) / 80.0 * LEVELS))
    t = k / (LEVELS - 1)
    lo, hi = (255, 255, 255), (23, 54, 110)
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(lo, hi))


def _panel(d, grid, x0, ticks, label):
    """grid[frame][row]: draw row 0 at the bottom, merging equal neighbours along time into one rectangle."""
    rows = len(grid[0])
    cell_h = PANEL_H / rows
    frames = len(grid)
    runs = {}
    for r in range(rows):
        t = 0
        while t < frames:
            shade = _shade(grid[t][r])
            u = t
            while u + 1 < frames and _shade(grid[u + 1][r]) == shade:
                u += 1
            runs.setdefault(shade, []).append((t, u, r))
            t = u + 1
    for shade, cells in runs.items():
        parts = []
        for t, u, r in cells:
            x = x0 + t * CELL_W
            y = GY + PANEL_H - (r + 1) * cell_h
            parts.append(f"M{x:.1f} {y:.1f}h{(u - t + 1) * CELL_W:.1f}v{cell_h:.2f}h-{(u - t + 1) * CELL_W:.1f}z")
        d.items.append(f'<path d="{"".join(parts)}" fill="{shade}"/>')
    d.items.append(f'<rect x="{x0}" y="{GY}" width="{frames * CELL_W:.1f}" height="{PANEL_H}" fill="none" stroke="#a0aec0"/>')
    for pos, text in ticks:
        y = GY + PANEL_H - pos * PANEL_H
        d.text(x0 - 6, y, [text], size=11, anchor="end")
    d.text(x0 + frames * CELL_W / 2, GY - 16, [label], size=13, weight="bold")
    for t in range(0, frames, 10):
        d.text(x0 + (t + 0.5) * CELL_W, GY + PANEL_H + 14, [str(t)], size=11)
    d.text(x0 + frames * CELL_W / 2, GY + PANEL_H + 32, ["frame t (hop 10 ms)"], size=12, color=MUTED)


def render() -> str:
    lin, mel = sp.figure_spectrograms()
    frames = len(lin)
    w = GX2 + frames * CELL_W + 20
    d = Diagram(w, GY + PANEL_H + 46)
    nyq = sp.FIG_RATE / 2
    _panel(d, lin, GX1, [(f / nyq, f"{f} Hz") for f in (0, 1000, 2000, 3000, 4000)], "65 linear bins, labelled by frequency")
    edges = sp.band_edges(sp.FIG_MELS, 0, nyq, "slaney")
    marks = [0, 6, 12, 18, 23]
    _panel(d, mel, GX2, [((m + 0.5) / sp.FIG_MELS, f"{edges[m + 1]:.0f} Hz") for m in marks], "24 mel filters, labelled by centre")
    return d.svg()
