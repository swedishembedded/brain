# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The omni model on one grid, in two panels: three inputs that become rows of one width and are read by one decoder, and the second
# decoder, the code predictor and the code-to-wave decoder that turn the reply into speech. Every size is a constant of `omni_models`
# (the configuration's widths, depths and expert counts), not typed.
#
# Swedish Embedded AB implements solutions for multimodal models that listen, look and speak. If your team needs expertise in
# omni-modal model engineering, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import omni_models as om  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 800, 560
BOX_H = 62


def render() -> str:
    d = Diagram(W, H)
    rows_pic = om._picture(om.MIN_H, om.MIN_W)[3]

    d.panel(20, 16, W - 40, 262, "Reading: three inputs become rows of 2,048 numbers")
    ys = (84, 150, 216)
    ins = [["16 kHz sound"], ["frames of video,", "pictures"], ["text tokens"]]
    towers = [["audio tower", f"{om.A_LAYERS} blocks, width {om.A_D:,}", f"{om._audio_rows(1)} rows a second"],
              ["vision tower", f"{om.V_DEPTH} blocks, width {om.V_D:,}", f"{rows_pic} rows a picture"],
              ["embedding table", f"{om.T_VOCAB:,} x {om.T_D:,}", "one row a token"]]
    kinds = ("plain", "plain", "plain")
    inbox, towbox = [], []
    for y, i, t, k in zip(ys, ins, towers, kinds):
        a = d.box(104, y, 130, BOX_H, i, k)
        b = d.box(300, y, 170, BOX_H, t, "model")
        d.arrow(a["r"], b["l"])
        inbox.append(a)
        towbox.append(b)
    think = d.box(640, 150, 220, 178, ["Thinker", f"{om.T_LAYERS} layers, width {om.T_D:,}", f"{om.T_EXPERTS} experts, {om.T_TOPK} used", "", "reads the rows in one", "sequence, writes text"], "result")
    for y, b in zip(ys, towbox):
        d.arrow(b["r"], (think["l"][0], y))

    d.panel(20, 294, W - 40, 250, "Answering: the reply is spoken by a second decoder")
    y0 = 470
    xs = (90, 245, 400, 555, 710)
    cw = 128
    reply = d.box(xs[0], y0, cw, BOX_H, ["reply tokens", "of the Thinker"], "result")
    talk = d.box(xs[1], y0, cw, BOX_H, ["Talker", f"{om.K_LAYERS} layers", f"{om.K_TOPK} of {om.K_EXPERTS} experts"], "model")
    pred = d.box(xs[2], y0, cw, BOX_H, ["code predictor", f"{om.M_LAYERS} layers", f"codes 1 to {om.NQ - 1}"], "model")
    wave = d.box(xs[3], y0, cw, BOX_H, ["Code2Wav", f"{om.C_LAYERS} layers, then", "upsamples x1,920"], "model")
    out = d.box(xs[4], y0, cw, BOX_H, ["speech", "24,000 samples", "a second"], "result")
    hid = d.box(xs[1], 396, 180, 44, ["layer-24 states of", "the media rows"], "accent")
    d.arrow(reply["r"], talk["l"])
    d.arrow(talk["r"], pred["l"])
    d.arrow(pred["r"], wave["l"])
    d.arrow(wave["r"], out["l"])
    d.arrow(hid["b"], talk["t"])
    d.path([(think["b"][0], think["b"][1]), (think["b"][0], 342), (xs[0], 342), (xs[0], reply["t"][1])], gap=3)
    return d.svg()
