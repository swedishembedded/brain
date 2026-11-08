# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The chain y = layer_norm(x + gelu(x W + b)) on an activation [2048, 1024] as five dispatches, as two (the matmul with bias,
# gelu and residual as its epilogue, then the layer norm) and as one. Every arrow between dispatches is an intermediate that is
# written to memory and read back; the byte counts are the worked module's `fusion_levels` and `chain_traffic`, so the picture
# and the tables of the chapter cannot disagree.
#
# Swedish Embedded AB implements solutions for removing the memory round trips between the small kernels of a machine learning
# workload. If your team needs expertise in kernel fusion for GPU inference and training, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fusion as fu  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 780, 452
COLS = [95 + 150 * i for i in range(5)]       # centres of the five columns
BOX_W, BOX_H = 96, 52
PANEL_Y = (14, 150, 286)
PANEL_H = 124
MIB = 1 << 20

TITLES = ("Five dispatches", "Two dispatches", "One dispatch")
NAMES = (["matmul", "x W"], ["bias", "+ b"], ["gelu"], ["residual", "+ x"], ["layer", "norm"])


def render() -> str:
    levels, boundary, _ = fu.figure_numbers()
    d = Diagram(W, H)
    for i, (label, n_disp, r, w) in enumerate(levels):
        d.panel(20, PANEL_Y[i], W - 40, PANEL_H, f"{TITLES[i]}: {(r + w) / MIB:.1f} MiB through memory")
    cy = [y + 82 for y in PANEL_Y]
    tag = f"{boundary / MIB:.0f} MiB"
    # panel 1: five boxes
    boxes = [d.box(x, cy[0], BOX_W, BOX_H, NAMES[k], "accent" if k == 0 else "model") for k, x in enumerate(COLS)]
    for k, name in enumerate(("z", "u", "g", "r")):
        d.arrow(boxes[k]["r"], boxes[k + 1]["l"], [name, tag])
    # panel 2: the epilogue-fused dispatch and the layer norm
    left, right = COLS[0] - BOX_W / 2, COLS[3] + BOX_W / 2
    fused = d.box((left + right) / 2, cy[1], right - left, BOX_H,
                  ["matmul + bias + gelu + residual", "z, u and g stay in registers"], "accent")
    ln = d.box(COLS[4], cy[1], BOX_W, BOX_H, NAMES[4], "model")
    d.arrow(fused["r"], ln["l"], ["r", tag])
    # panel 3: everything in one dispatch
    left, right = COLS[0] - BOX_W / 2, COLS[4] + BOX_W / 2
    d.box((left + right) / 2, cy[2], right - left, BOX_H,
          ["matmul + bias + gelu + residual + layer norm", "reads x, W, b, gamma, beta once and writes y once"], "accent")
    # legend
    ly = H - 18
    d.swatch(40, ly - 8, 22, 14, "accent")
    d.text(70, ly, ["compute-bound dispatch (intensity above the ridge point)"], size=12, anchor="start")
    d.swatch(430, ly - 8, 22, 14, "model")
    d.text(460, ly, ["bandwidth-bound dispatch"], size=12, anchor="start")
    return d.svg()
