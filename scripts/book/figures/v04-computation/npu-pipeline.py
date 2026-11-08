# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The path of a network to an accelerator: model, exported graph, quantized graph, command stream, device. Each box names
# the artifact and the size of the chapter's tiny network at that stage (the worked module's `pipeline_sizes`); the line
# under each box says what that stage fixes, so a reader sees which decisions are taken before the device ever runs.
#
# Swedish Embedded AB implements solutions for compiling whole neural networks into programs for fixed-function
# accelerators. If your team needs expertise in graph export, post-training quantization and NPU compilers, you can
# procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import npu  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 860, 244
COLS = [90 + 170 * i for i in range(5)]
BOX_W, BOX_H = 120, 78
Y_BOX = 100
Y_FIX = 196


def render() -> str:
    s = npu.pipeline_sizes()
    d = Diagram(W, H)
    d.panel(20, 14, W - 40, H - 28, "One network, five forms")
    stages = [
        (["model", "float weights", "and the code"], "plain"),
        (["exported graph", "static shapes", f"{s['fp32']} B of f32"], "model"),
        (["quantized graph", "int8 and int32", f"{s['quantized']} B + ranges"], "model"),
        (["command stream", f"{s['tasks']} tasks of {s['task']:,} B", f"{s['stream']:,} B"], "accent"),
        (["device", "runs the stream", "with no host code"], "result"),
    ]
    boxes = [d.box(x, Y_BOX, BOX_W, BOX_H, lines, kind) for x, (lines, kind) in zip(COLS, stages)]
    for a, b, verb in zip(boxes, boxes[1:], (["export"], ["quantize"], ["compile"], ["submit"])):
        d.arrow(a["r"], b["l"], verb)
    fixed = [["nothing yet"], ["operators,", "order, shapes"], ["ranges, scales,", "zero points, codes"],
             ["fusion, tiles,", "addresses, words"], ["only the values", "of the input"]]
    d.text(40, Y_FIX - 36, ["decided at this stage:"], size=12, color="#4a5568", anchor="start")
    for x, lines in zip(COLS, fixed):
        d.text(x, Y_FIX, lines, size=12, color="#1a202c")
    return d.svg()
