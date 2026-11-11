# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The project's closed loop. Top lane: the waveform of a sentence is resampled from 12 kHz to 8 kHz, framed into 50 log-mel frames
# of 16 values a second, (optionally) quantized, and read by a CTC recogniser. Bottom lane: a reference clip's embedding gives an
# estimated speaker parameter, and the synthesiser speaks the recogniser's transcript with it. The rates and widths are the module
# `project_speech_pipeline`'s constants, not typed.
#
# Swedish Embedded AB implements solutions for speech pipelines whose rates, front ends and error budgets are specified and tested
# end to end. If your team needs expertise in speech recognition and synthesis engineering, you can procure our services by sending
# an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_speech_pipeline as ps  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, GAP, TOP = 800, 18, 16
COLS = (80, 210, 340, 470, 600, 730)
BOX_W, BOX_H = 112, 70
H1, H2 = 140, 190


def render() -> str:
    fps = ps.FS_REC // ps.HOP
    d = Diagram(W, TOP + H1 + GAP + H2 + 16)

    y = TOP
    d.panel(20, y, W - 40, H1, "Listening: a waveform becomes text")
    cy = y + 88
    boxes = [
        (["waveform", f"{ps.FS_SYN:,} / s"], "plain"),
        (["Kaiser sinc", f"resampler", f"{ps.FS_REC:,} / s"], "model"),
        (["log-mel", f"{fps} frames / s", f"x {ps.N_MELS}, standardized"], "model"),
        (["residual", "quantizer", f"{ps.RVQ_Q} x {ps.RVQ_K} codewords"], "accent"),
        ([f"CTC recogniser", f"{ps.D_IN} - {ps.HID} - {ps.K}", "greedy decode"], "model"),
        (["text", "letters and", "spaces"], "result"),
    ]
    placed = [d.box(COLS[i], cy, BOX_W, BOX_H, text, kind) for i, (text, kind) in enumerate(boxes)]
    for a, b in zip(placed, placed[1:]):
        d.arrow(a["r"], b["l"])

    y = TOP + H1 + GAP
    d.panel(20, y, W - 40, H2, "Speaking: the transcript is spoken again in the voice of a reference clip")
    top, bot, mid = y + 74, y + 146, y + 110
    ref = d.box(COLS[0], top, BOX_W, BOX_H - 14, ["reference clip", f"{ps.ENROL} clips enrol"], "plain")
    emb = d.box(COLS[1], top, BOX_W, BOX_H - 14, ["embedding", f"{ps.N_MELS} values, unit"], "model")
    rid = d.box(COLS[2], top, BOX_W, BOX_H - 14, ["ridge map", "speaker g"], "model")
    txt = d.box(COLS[2], bot, BOX_W, BOX_H - 14, ["transcript", "from the recogniser"], "plain")
    syn = d.box(COLS[3] + 30, mid, BOX_W + 10, BOX_H - 6, ["synthesiser", "2 sinusoids a letter", "scaled by g"], "model")
    out = d.box(COLS[5] - 30, mid, BOX_W + 10, BOX_H - 14, ["generated audio", f"{ps.FS_SYN:,} / s"], "result")
    d.arrow(ref["r"], emb["l"])
    d.arrow(emb["r"], rid["l"])
    d.arrow(rid["r"], syn["t"])
    d.arrow(txt["r"], syn["b"])
    d.arrow(syn["r"], out["l"])
    return d.svg()
