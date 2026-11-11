# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The log-mel frames of one test sentence of the project's world, twice: after the Kaiser-tapered resampler and after linear
# interpolation. The letter d carries a 5200 Hz component that the 8 kHz recogniser must not hear; the unfiltered resampler folds
# it to 8000 minus 5200 g, 3424 Hz for the speaker of this sentence (g = 0.88), a dark mark in the top band that the correct front end lacks. Darker is stronger, in decibels below the
# largest cell of the sentence. The data are `project_speech_pipeline.figure_mels`.
#
# Swedish Embedded AB implements solutions for speech front ends whose resampling and mel conventions are tested before they ship. If
# your team needs expertise in audio front ends for speech models, you can procure our services by sending an email to
# info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_speech_pipeline as ps  # noqa: E402
import spectrograms as sp  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

CELL_W = 6.0
PANEL_H = 176.0
LEVELS = 9
FLOOR_DB = 40.0
GX1 = 64
GY = 52


def _shade(db):
    """White at -40 dB, dark blue at 0 dB, in LEVELS steps."""
    k = min(LEVELS - 1, max(0, int((db + FLOOR_DB) / FLOOR_DB * LEVELS)))
    t = k / (LEVELS - 1)
    lo, hi = (255, 255, 255), (23, 54, 110)
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(lo, hi))


def _panel(d, grid, x0, ticks, label):
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
        d.text(x0 - 6, GY + PANEL_H - pos * PANEL_H, [text], size=11, anchor="end")
    d.text(x0 + frames * CELL_W / 2, GY - 16, [label], size=13, weight="bold")
    for t in range(0, frames, 10):
        d.text(x0 + (t + 0.5) * CELL_W, GY + PANEL_H + 14, [str(t)], size=11)
    d.text(x0 + frames * CELL_W / 2, GY + PANEL_H + 32, ["frame t (hop 20 ms)"], size=12, color=MUTED)


def _db(frames):
    """Decibels below the largest value. The frames are (ln E - largest) / 4, so one unit is 4 nepers of energy, 4 * 10 / ln 10 decibels."""
    top = max(max(r) for r in frames)
    return [[(v - top) * 4.0 * 10.0 / math.log(10.0) for v in r] for r in frames]


def render() -> str:
    text, clean, lin = ps.figure_mels()
    frames = len(clean)
    gx2 = GX1 + frames * CELL_W + 56
    d = Diagram(gx2 + frames * CELL_W + 20, GY + PANEL_H + 46)
    edges = sp.band_edges(ps.N_MELS, 0.0, 4000.0, "slaney")
    marks = [0, 4, 8, 12, 15]
    ticks = [((m + 0.5) / ps.N_MELS, f"{edges[m + 1]:.0f} Hz") for m in marks]
    _panel(d, _db(clean), GX1, ticks, f"Kaiser sinc, sentence: {text}")
    _panel(d, _db(lin), gx2, ticks, "linear interpolation")
    return d.svg()
