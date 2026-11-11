# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two directions of brain's 12.5 Hz codec on one grid, each stage labelled with the rate at which it produces columns and the
# width of a column. The rates are the arithmetic of `audio_codecs` (24,000 Hz divided by the strides 4, 5, 6, 8 and 2), not typed.
#
# Swedish Embedded AB implements solutions for turning audio into discrete tokens that a language model can read and write. If your
# team needs expertise in neural audio codecs, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import audio_codecs as ac  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, PANEL_H, GAP, TOP = 780, 130, 18, 16
COLS = (100, 248, 396, 544, 692)
BOX_W, BOX_H = 124, 66


def _hz(x):
    return f"{x:g} Hz"


def render() -> str:
    r = ac.pipeline_rates()
    d = Diagram(W, TOP + 2 * PANEL_H + GAP + 16)

    y = TOP
    d.panel(20, y, W - 40, PANEL_H, "Encode: one clip in, T frames of 16 integers out")
    cy = y + 82
    boxes = [
        (["waveform", f"{_hz(r['samples'])} x 1"], "plain"),
        (["strided convs", "4, 5, 6, 8", f"{_hz(r['convolutions'])} x {ac.HIDDEN}"], "model"),
        (["transformer", f"{_hz(r['convolutions'])} x {ac.HIDDEN}"], "model"),
        (["strided conv", "by 2", f"{_hz(r['frames'])} x {ac.HIDDEN}"], "model"),
        (["residual quantizer", f"{_hz(r['frames'])} x {ac.NQ} codes"], "result"),
    ]
    placed = [d.box(COLS[i], cy, BOX_W, BOX_H, text, kind) for i, (text, kind) in enumerate(boxes)]
    for a, b in zip(placed, placed[1:]):
        d.arrow(a["r"], b["l"])

    y = TOP + PANEL_H + GAP
    d.panel(20, y, W - 40, PANEL_H, "Decode: 16 integers per frame in, a clip out")
    cy = y + 82
    boxes = [
        (["16 codes", f"{_hz(r['frames'])}"], "result"),
        (["gather, sum,", "project", f"{_hz(r['frames'])} x {ac.LAT}"], "model"),
        (["conv, transformer", f"{_hz(r['frames'])} x {ac.LATENT}"], "model"),
        (["transposed convs", "2, 2", f"{_hz(r['frames'] * 4)} x {ac.LATENT}"], "model"),
        (["transposed convs", "8, 5, 4, 3", f"{_hz(r['samples'])} x 1"], "model"),
    ]
    placed = [d.box(COLS[i], cy, BOX_W, BOX_H, text, kind) for i, (text, kind) in enumerate(boxes)]
    for a, b in zip(placed, placed[1:]):
        d.arrow(a["r"], b["l"])
    return d.svg()
