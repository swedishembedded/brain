# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The five components of the song generator on one grid, in two panels: the part that writes the whole song one frame at a time, and
# the part that paints it in chunks of eight seconds. Every rate and size is a constant of `music` (25 frames, 8 hidden vectors of
# 4096, 689-step chunks, 44,100 Hz), not typed.
#
# Swedish Embedded AB implements solutions for generating long audio from a description. If your team needs expertise in music and
# long-form audio generation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import music as mu  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, PANEL_H, GAP, TOP = 780, 196, 18, 16
COLS = (100, 248, 396, 544, 692)
BOX_W, BOX_H = 124, 66
SMALL_H = 46


def render() -> str:
    d = Diagram(W, TOP + 2 * PANEL_H + GAP + 16)
    chunk = mu.latent_length(mu.CHUNK_FRAMES)

    y = TOP
    d.panel(20, y, W - 40, PANEL_H, f"Writing the whole song, one frame at a time ({mu.FRAME_RATE} frames a second)")
    top, main = y + 78, y + 150
    text = d.box(COLS[0], main, BOX_W, BOX_H, ["caption and", "lyrics", "as tokens"], "plain")
    lm = d.box(COLS[1], main, BOX_W, BOX_H, ["Global LLM", f"{mu.LM['L']} layers", "1 semantic code"], "model")
    cfg = d.box(COLS[1], top, BOX_W, SMALL_H, ["second branch:", "prompt masked"], "accent")
    dd = d.box(COLS[2], main, BOX_W, BOX_H, ["depth decoder", f"{mu.DD['layers']} layers", f"{mu.N_RES} residual codes"], "model")
    hid = d.box(COLS[3], main, BOX_W, BOX_H, [f"{mu.N_HID} hidden vectors", f"of {mu.CE['hidden']} per frame", "kept, not the codes"], "accent")
    end = d.box(COLS[4], main, BOX_W, BOX_H, ["end token", "or the frame", "cap"], "result")
    d.arrow(text["r"], lm["l"])
    d.arrow(cfg["b"], lm["t"])
    d.arrow(lm["r"], dd["l"])
    d.arrow(dd["r"], hid["l"])
    d.arrow(hid["r"], end["l"])

    y = TOP + PANEL_H + GAP
    d.panel(20, y, W - 40, PANEL_H, f"Painting it in chunks of {chunk} latent steps, {mu.CHUNK_FRAMES} frames each")
    top, main = y + 78, y + 150
    cond = d.box(COLS[0], main, BOX_W, BOX_H, ["condition", "encoder", f"{mu.N_HID} x {mu.CE['hidden']} to {mu.CE['out']}"], "model")
    dit = d.box(COLS[1], main, BOX_W, BOX_H, ["flow DiT", f"{mu.DIT['layers']} blocks, {mu.STEPS} steps", "2 guidance passes"], "model")
    carry = d.box(COLS[1], top, BOX_W, SMALL_H, [f"{mu.OVERLAP} steps carried", "from the last chunk"], "accent")
    voc = d.box(COLS[2], main, BOX_W, BOX_H, ["vocoder", f"x{mu.HOP_LAT} upsampling", "two streams"], "model")
    crop = d.box(COLS[3], main, BOX_W, BOX_H, ["crop and", f"join: {mu.CROP_L} off the left,", f"{mu.CROP_R} off the right"], "model")
    wav = d.box(COLS[4], main, BOX_W, BOX_H, ["waveform", f"stereo, {mu.SR:,} Hz"], "result")
    d.arrow(cond["r"], dit["l"])
    d.arrow(carry["b"], dit["t"])
    d.arrow(dit["r"], voc["l"])
    d.arrow(voc["r"], crop["l"])
    d.arrow(crop["r"], wav["l"])
    return d.svg()
