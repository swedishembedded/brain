# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The decoder block of GPT-2 small beside that of Qwen3 0.6B, row against row: what each stage of the block is in the two models.
# The yellow box is the one stage that GPT-2 has no counterpart for. Parameter counts come from `modern_decoder.parameters`, the
# function behind the tables of the chapter.
#
# Swedish Embedded AB implements solutions for deploying open-weight language models, from the architecture to the checkpoint. If
# your team needs expertise in decoder architectures, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import modern_decoder as md  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 900, 690
PANEL_W, PANEL_H = 420, 670
LEFT_X, RIGHT_X = 15, 465
BOX_W, BOX_H = 340, 48
ROWS = (82, 156, 230, 304, 378, 452, 526, 600)   # centres of the eight stages


def column(d, x0, title, rows, layers, total):
    d.panel(x0, 10, PANEL_W, PANEL_H, title)
    cx = x0 + PANEL_W / 2
    boxes = []
    for y, (lines, kind) in zip(ROWS, rows):
        boxes.append(d.box(cx, y, BOX_W, BOX_H, lines, kind))
    for a, b in zip(boxes, boxes[1:]):
        d.arrow(a["b"], b["t"])
    top = ROWS[1] - BOX_H / 2 - 20
    bottom = ROWS[6] + BOX_H / 2 + 20
    d.group(cx - BOX_W / 2 - 14, top, BOX_W + 28, bottom - top)
    d.text(cx + BOX_W / 2 + 6, top + 10, [f"x {layers} layers"], size=12, color=MUTED, anchor="end")
    d.text(cx, ROWS[7] + 52, [f"{md._int(total)} parameters"], size=13, color=MUTED)


def render() -> str:
    a, b = md.GPT2, md.QWEN3_0_6B
    d = Diagram(W, H)
    left = [(["token embedding + learned position", f"({md._int(a['pos'])} rows added at the input)"], "accent"),
            (["LayerNorm (gain and shift)"], "model"),
            (["Q, K, V projections with biases", f"{a['H']} heads, one key-value head each"], "model"),
            (["no step between projection and scores", "(positions entered at the input)"], "plain"),
            (["causal attention, then W_O with bias", "and the residual add"], "model"),
            (["LayerNorm (gain and shift)"], "model"),
            (["GELU network 768 - 3,072 - 768,", "with biases, and the residual add"], "model"),
            (["LayerNorm, then the head,", "tied to the embedding table"], "model")]
    right = [(["token embedding", "(no position table)"], "accent"),
             (["RMS norm (gain only)"], "model"),
             (["Q, K, V projections, no biases", f"{b['H']} query heads, {b['hkv']} key-value heads"], "model"),
             (["QK-norm per head, then the", "rotary rotation by position"], "result"),
             (["causal attention, each key-value head", "read by 2 query heads; W_O, residual add"], "model"),
             (["RMS norm (gain only)"], "model"),
             (["SwiGLU 1,024 - 3,072 - 1,024,", "no biases, and the residual add"], "model"),
             (["RMS norm, then the head: tied here,", "a separate matrix at 8B"], "model")]
    column(d, LEFT_X, "GPT-2 small (2019)", left, a["L"], md.parameters(a))
    column(d, RIGHT_X, "Qwen3 0.6B as brain implements it", right, b["L"], md.parameters(b))
    return d.svg()
