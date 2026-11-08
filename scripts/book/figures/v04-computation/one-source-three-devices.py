# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The paths of one kernel source to three kinds of device: a Vulkan-style GPU (the source is parsed and checked once, written
# as SPIR-V and compiled by the driver), a CUDA GPU (the same checked form is written as CUDA C++ text and compiled by NVRTC at
# run time, or loaded from a cubin built ahead of time) and a CPU (the same form is compiled to machine code by a JIT and the
# kernel becomes loops). The counts under the devices are the worked module's `backends.three_paths` runs of the kernel
# y = 2 x + 1 over 10 numbers, so the picture and the table of the chapter cannot disagree.
#
# Swedish Embedded AB implements solutions for running one set of compute kernels on every device a client owns. If your team
# needs expertise in shader translation and kernel compilation, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import backends as bk  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 900, 450
C = (85, 245, 430, 625, 805)          # centres of the five columns
ROWS = (100, 220, 340)                # centres of the three device rows
BW, BH = 142, 56


def render() -> str:
    x = bk._x_input()
    _, groups, ran = bk._run_invocation_model(x, list(range(12)))
    _, blocks, ran_b = bk._run_cuda_model(x, [bk.N_AXPB, bk._bits(bk.SCALE), bk._bits(bk.SHIFT)])
    _, loops, ran_c = bk._run_cpu_model(x)
    d = Diagram(W, H)
    for cx, title in zip(C, ("source", "front end", "text or IR for the target", "compiler, at run time", "device")):
        d.text(cx, 22, [title], size=12, color="#4a5568", weight="bold")
    src = d.box(C[0], ROWS[1], 124, BH, ["WGSL kernel", "one text"], "accent")
    fe = d.box(C[1], ROWS[1], BW, BH, ["parse, validate", "one checked IR"], "model")
    d.arrow(src["r"], fe["l"], None)
    targets = (
        (["SPIR-V words", "binary IR"], ["driver compiler", "builds a pipeline"], ["Vulkan GPU", f"{groups} workgroups x {bk.WG_AXPB}", f"{ran} invocations"]),
        (["CUDA C++ text", "indices clamped"], ["NVRTC compiles", "to a cubin"], ["CUDA GPU", f"{blocks} blocks x {bk.WG_AXPB}", f"{ran_b} threads"]),
        (["native-compiler IR", "per-invocation code"], ["JIT compiler", "to machine code"], ["CPU cores", f"{loops} x {bk.WG_AXPB} loop nest", f"{ran_c} body runs"]),
    )
    trunk = fe["r"][0] + 28
    for row, (tgt, comp, dev) in zip(ROWS, targets):
        t = d.box(C[2], row, BW, BH, tgt, "model")
        c = d.box(C[3], row, BW, BH, comp, "accent")
        v = d.box(C[4], row, BW, BH, dev, "result")
        d.arrow(t["r"], c["l"], None)
        d.arrow(c["r"], v["l"], None)
        if row == ROWS[1]:
            d.arrow(fe["r"], t["l"], None)
        else:
            d.path([fe["r"], (trunk, fe["r"][1]), (trunk, row), t["l"]], radius=10)
    aot = d.box(C[3], 280, BW, 34, ["or a cubin built", "ahead of time"], "plain")
    d.line((C[3], ROWS[1] + BH / 2), (C[3], 280 - 17))
    ly = H - 20
    d.swatch(40, ly - 8, 22, 14, "model")
    d.text(70, ly, ["text or IR the target's tools take"], size=12, anchor="start")
    d.swatch(330, ly - 8, 22, 14, "accent")
    d.text(360, ly, ["translator or compiler stage"], size=12, anchor="start")
    d.swatch(590, ly - 8, 22, 14, "result")
    d.text(620, ly, ["the device runs the kernel"], size=12, anchor="start")
    _ = aot
    return d.svg()
