# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two speech synthesisers of brain on one grid, each stage labelled with the rate at which it produces steps and what a step
# is, and the two places where a voice enters. The rates are the constants of `speech_synthesis` (12.5 codec frames, 25 speech
# tokens, 50 mel frames, 24,000 samples a second), not typed.
#
# Swedish Embedded AB implements solutions for turning text into speech in a chosen voice. If your team needs expertise in neural
# text-to-speech, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import speech_synthesis as ss  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, PANEL_H, GAP, TOP = 780, 196, 18, 16
COLS = (100, 248, 396, 544, 692)
BOX_W, BOX_H = 124, 66
SMALL_H = 46


def render() -> str:
    d = Diagram(W, TOP + 2 * PANEL_H + GAP + 16)

    y = TOP
    d.panel(20, y, W - 40, PANEL_H, "Codec synthesiser: text to 16 codes per frame, then audio")
    top, main = y + 78, y + 150
    talker = d.box(COLS[1], main, BOX_W, BOX_H, ["Talker", "writes code 0", f"{ss.FRAME_RATE:g} frames/s"], "model")
    spk = d.box(COLS[1], top, BOX_W, SMALL_H, ["speaker vector", "1024 numbers"], "accent")
    text = d.box(COLS[0], main, BOX_W, BOX_H, ["text tokens", "language tag"], "plain")
    pred = d.box(COLS[2], main, BOX_W, BOX_H, ["code predictor", "writes codes 1 .. 15", "16 small steps"], "model")
    dec = d.box(COLS[3], main, BOX_W, BOX_H, ["codec decoder", f"{ss.NQ} codes to", f"{ss.SAMPLES_PER_FRAME} samples"], "model")
    wav = d.box(COLS[4], main, BOX_W, BOX_H, ["waveform", f"{ss.FS:,} Hz"], "result")
    d.arrow(text["r"], talker["l"])
    d.arrow(spk["b"], talker["t"])
    d.arrow(talker["r"], pred["l"])
    d.arrow(pred["r"], dec["l"])
    d.arrow(dec["r"], wav["l"])

    y = TOP + PANEL_H + GAP
    d.panel(20, y, W - 40, PANEL_H, "Speech-token synthesiser: text to tokens, to a mel spectrogram, to audio")
    top, main = y + 78, y + 150
    text = d.box(COLS[0], main, BOX_W, BOX_H, ["text tokens", "reference text"], "plain")
    ref = d.box(COLS[1], top, BOX_W, SMALL_H, ["reference tokens", f"{ss.S_RATE} per second"], "accent")
    lm = d.box(COLS[1], main, BOX_W, BOX_H, ["speech-token LM", "24 layers", f"{ss.S_RATE} tokens/s"], "model")
    spk = d.box(COLS[2], top, BOX_W, SMALL_H, ["speaker vector", "192 -> 80 numbers"], "accent")
    flow = d.box(COLS[2], main, BOX_W, BOX_H, ["flow decoder", "10 Euler steps", f"{ss.MEL_RATE} x {ss.MEL_BINS} mel/s"], "model")
    voc = d.box(COLS[3], main, BOX_W, BOX_H, ["vocoder", "mel to samples", f"{ss.FS:,} Hz"], "model")
    wav = d.box(COLS[4], main, BOX_W, BOX_H, ["waveform", f"{ss.FS:,} Hz"], "result")
    d.arrow(text["r"], lm["l"])
    d.arrow(ref["b"], lm["t"])
    d.arrow(lm["r"], flow["l"])
    d.arrow(spk["b"], flow["t"])
    d.arrow(flow["r"], voc["l"])
    d.arrow(voc["r"], wav["l"])
    return d.svg()
