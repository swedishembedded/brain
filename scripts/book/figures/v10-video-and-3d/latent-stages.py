# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The shape of a 121-frame, 512 by 768 clip after each stage of a causal video autoencoder's encoder. The shapes are the arithmetic of
# `video.encoder_stages` (patch size 4, then down blocks that halve space, time, and both), not typed.
#
# Swedish Embedded AB implements solutions for compressing video into the latent grid a generative model works on. If your team
# needs expertise in video autoencoders, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import video  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, PANEL_H, TOP = 800, 150, 16
BOX_W, BOX_H = 100, 84
LABELS = ["video", "patch 4", "space down", "time down", "all down", "all down", "latent"]
KINDS = ["plain", "model", "model", "model", "model", "model", "result"]


def render() -> str:
    st = video.encoder_stages(121, 512, 768)
    st = st[:-1]  # the mean channels of the last stage are the latent; the spread channel is not carried on
    st[-1:] = [st[-1], ("latent", video.LATENT_C, st[-1][2], st[-1][3], st[-1][4])]
    d = Diagram(W, TOP + PANEL_H + 12)
    d.panel(20, TOP, W - 40, PANEL_H, "Encoder: channels x frames, then height x width, after each stage")
    cy = TOP + 92
    cols = [76 + 108 * i for i in range(len(st))]
    placed = []
    for i, (_, c, t, h, w) in enumerate(st):
        placed.append(d.box(cols[i], cy, BOX_W, BOX_H, [LABELS[i], f"{c} x {t}", f"{h} x {w}"], KINDS[i]))
    for a, b in zip(placed, placed[1:]):
        d.arrow(a["r"], b["l"])
    return d.svg()
