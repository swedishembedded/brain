# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two speech recognisers of brain on one grid, each stage labelled with the rate at which it produces rows and the width of a row.
# The rates and widths are those of `speech_recognition` (frames at 16,000 / 160 a second, 13 rows a second for the chunked encoder of
# Qwen3-ASR, 12.5 for the Nemotron encoder, which is 100 / 8), not typed.
#
# Swedish Embedded AB implements solutions for speech-to-text systems whose rates, latency and memory are counted before they are
# deployed. If your team needs expertise in speech recognition engineering, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import speech_recognition as sr  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, PANEL_H, GAP, TOP = 880, 140, 18, 16
COLS = (85, 225, 365, 505, 645, 785)
BOX_W, BOX_H = 120, 70


def render() -> str:
    r = sr.pipeline_rates()
    d = Diagram(W, TOP + 2 * PANEL_H + GAP + 16)

    y = TOP
    d.panel(20, y, W - 40, PANEL_H, "Qwen3-ASR: a decoder that reads the audio rows as part of its prompt")
    cy = y + 88
    boxes = [
        (["waveform", "16,000 / s"], "plain"),
        (["log-mel", f"{r['qwen_frames']} / s x 128"], "model"),
        (["3 strided convs", f"{r['qwen_rows']} rows / s", f"x {r['qwen_width']}"], "model"),
        ([f"{r['qwen_layers']} blocks", "windows of", "104 rows"], "model"),
        (["projector", f"{r['qwen_rows']} rows / s", f"x {r['qwen_out']}"], "model"),
        ([f"decoder, {r['qwen_dec_layers']} layers", "writes the", "text"], "result"),
    ]
    placed = [d.box(COLS[i], cy, BOX_W, BOX_H, text, kind) for i, (text, kind) in enumerate(boxes)]
    for a, b in zip(placed, placed[1:]):
        d.arrow(a["r"], b["l"])

    y = TOP + PANEL_H + GAP
    d.panel(20, y, W - 40, PANEL_H, "Nemotron: a transducer that writes at most a few letters at each row")
    cy = y + 88
    boxes = [
        (["waveform", "16,000 / s"], "plain"),
        (["log-mel", f"{r['nem_frames']} / s x 128"], "model"),
        (["3 strided convs", f"{r['nem_rows']:g} rows / s", f"x {r['nem_width']}"], "model"),
        ([f"{r['nem_layers']} Conformer", "blocks, chunks", "of 4 rows"], "model"),
        (["projectors", f"{r['nem_rows']:g} rows / s", f"x {r['nem_dec']}"], "model"),
        (["predictor and", "joint: blank", "or a piece"], "result"),
    ]
    placed = [d.box(COLS[i], cy, BOX_W, BOX_H, text, kind) for i, (text, kind) in enumerate(boxes)]
    for a, b in zip(placed, placed[1:]):
        d.arrow(a["r"], b["l"])
    return d.svg()
