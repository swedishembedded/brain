# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two speaker encoders of brain on one grid, each stage labelled with the rate at which it produces rows and the width of a row.
# The rates and widths are those of `speaker_embeddings` (24,000 / 256 = 93.75 frames a second and 16,000 / 160 = 100, halved by the
# stride-2 convolution of the second encoder, and the widths of the configurations), not typed.
#
# Swedish Embedded AB implements solutions for recognising who is speaking from a few seconds of audio, with the cost of the encoder
# counted before it is deployed. If your team needs expertise in speaker verification and voice-conditioned speech synthesis, you can
# procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import speaker_embeddings as se  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, PANEL_H, GAP, TOP = 800, 140, 18, 16
COLS = (75, 185, 295, 405, 515, 625, 735)
BOX_W, BOX_H = 100, 70


def render() -> str:
    r = se.figure_encoders()
    d = Diagram(W, TOP + 2 * PANEL_H + GAP + 16)

    y = TOP
    d.panel(20, y, W - 40, PANEL_H, "Encoder 1 (ECAPA-TDNN): the speaker vector that a voice-cloning synthesiser reads as one position")
    cy = y + 88
    boxes = [
        (["waveform", "24,000 / s"], "plain"),
        (["log-mel", f"{r['e_frames']:g} / s", f"x {r['e_mel']}"], "model"),
        (["time-delay", "convolution", f"k 5, x {r['e_width']}"], "model"),
        (["3 SE-Res2Net", "blocks, d 2 3 4", f"x {r['e_width']} each"], "model"),
        (["concatenate", "and convolve", f"x {r['e_agg']}"], "model"),
        (["attentive", "statistics", f"x {r['e_pool']}"], "model"),
        (["projection", f"{r['e_emb']} numbers", "per clip"], "result"),
    ]
    placed = [d.box(COLS[i], cy, BOX_W, BOX_H, text, kind) for i, (text, kind) in enumerate(boxes)]
    for a, b in zip(placed, placed[1:]):
        d.arrow(a["r"], b["l"])

    y = TOP + PANEL_H + GAP
    d.panel(20, y, W - 40, PANEL_H, "Encoder 2 (CAM++): the speaker vector that a flow-matching decoder reads after a normalisation")
    cy = y + 88
    boxes = [
        (["waveform", "16,000 / s"], "plain"),
        (["filterbank", f"{r['c_frames']:g} / s", f"x {r['c_mel']}"], "model"),
        (["2-D stem", "frequency /8", f"x {r['c_stem']}"], "model"),
        (["time-delay", "conv, stride 2", f"{r['c_rows']:g} / s x {r['c_width']}"], "model"),
        ([f"{r['c_layers']} dense", "context-gated", "layers"], "model"),
        (["statistics", "mean and std", f"x {r['c_pool']}"], "model"),
        (["projection", f"{r['c_emb']} numbers", "per clip"], "result"),
    ]
    placed = [d.box(COLS[i], cy, BOX_W, BOX_H, text, kind) for i, (text, kind) in enumerate(boxes)]
    for a, b in zip(placed, placed[1:]):
        d.arrow(a["r"], b["l"])
    return d.svg()
