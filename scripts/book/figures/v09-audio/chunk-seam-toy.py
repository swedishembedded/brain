# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The texture (song minus melody) of channel 0 of the chunked flow toy of the chapter, stitched from four chunks, once with the chunks
# denoised independently and once with each chunk's prefix held to the slice the previous chunk carried. The curves are
# `music.seam_figure_data()`: the same draws as the chapter's table. Dashed lines mark the seams (where two kept spans meet) and the
# joins (where the free part of a chunk begins after its held prefix).
#
# Swedish Embedded AB implements solutions for generating long audio in overlapping windows. If your team needs expertise in music
# and long-form audio generation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import music as mu  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curves, seams, joins = mu.seam_figure_data()
    n = mu.TOY["n_song"]
    lo = min(min(v) for v in curves.values()) - 0.15
    hi = max(max(v) for v in curves.values()) + 0.75
    p = Plot((0, n - 1), (lo, hi), width=600, height=320, xlabel="position in the song (latent steps)",
             ylabel="texture of channel 0", title="Stitching four chunks", legend_loc="below")
    xs = list(range(n))
    for s in seams:
        p.line([s - 0.5] * 2, [lo, hi - 0.3], color="#2f855a", width=1.2, dash=True)
    for j in joins:
        p.line([j - 0.5] * 2, [lo, hi - 0.3], color="#c05621", width=1.2, dash=True)
    p.line(xs, curves[False], color=3, label="independent chunks", width=2.2)
    p.line(xs, curves[True], color=0, label="prefix held", width=2.2)
    p.text(seams[0] - 0.5, hi - 0.12, "seam", anchor="end", size=11, color="#2f855a")
    p.text(joins[0] - 0.5, hi - 0.12, "join", anchor="start", size=11, color="#c05621")
    return p.svg()
