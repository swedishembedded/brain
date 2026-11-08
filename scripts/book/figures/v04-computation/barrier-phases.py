# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One workgroup of 4 invocations of the kernel `wg_sum` (workgroup 1 of the worked example `backends.cpu_phases`: x = 5, 6, 7, 8)
# as a GPU runs it (each phase is executed by all 4 invocations together, the barrier is a rendezvous) and as a CPU compiler
# writes it (each phase is a loop over the 4 invocations, the barrier is the end of the first loop). The numbers are the
# worked module's `phase_figure_numbers`.
#
# Swedish Embedded AB implements solutions for compiling GPU kernels to CPU code that keeps the guarantees of the kernel
# language. If your team needs expertise in kernel compilation, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import backends as bk  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, H = 900, 440
PW, PH = 420, 410
PX = (20, 460)
BOX_W, BOX_H = 82, 46
OFF = (60, 160, 260, 360)            # box centres inside a panel
Y_ONE, Y_BAR, Y_TWO, Y_SH = 108, 178, 248, 330


def render() -> str:
    f = bk.phase_figure_numbers()
    shared, total, n = f["shared"], f["sum"], f["wg"]
    d = Diagram(W, H)
    titles = ("GPU: the 4 invocations run each phase together", "CPU: each phase is a loop over the 4 invocations")
    for p, (px, title) in enumerate(zip(PX, titles)):
        d.panel(px, 14, PW, PH, title)
        row1, row2 = [], []
        for l in range(n):
            cx = px + OFF[l]
            row1.append(d.box(cx, Y_ONE, BOX_W, BOX_H, [f"l = {l}", f"s[{l}] = {shared[l]:g}"], "model"))
            lines = [f"l = {l}", f"out = {total:g}"] if l == 0 else [f"l = {l}", "idle"]
            row2.append(d.box(cx, Y_TWO, BOX_W, BOX_H, lines, "accent" if l == 0 else "plain"))
        if p == 0:
            d.group(px + OFF[0] - 52, Y_ONE - 34, OFF[3] - OFF[0] + 104, 68)
            d.group(px + OFF[0] - 52, Y_TWO - 34, OFF[3] - OFF[0] + 104, 68)
            d.text(px + PW / 2, 52, ["phase 1: all at once"], size=12, color="#4a5568")
            d.text(px + PW / 2, 295, ["phase 2: all at once"], size=12, color="#4a5568")
            bar = "barrier: all 4 have arrived"
        else:
            for l in range(n - 1):
                d.arrow(row1[l]["r"], row1[l + 1]["l"], None, gap=1)
                d.arrow(row2[l]["r"], row2[l + 1]["l"], None, gap=1)
            d.text(px + PW / 2, 52, ["loop 1: for l in 0 .. 4, in order"], size=12, color="#4a5568")
            d.text(px + PW / 2, 295, ["loop 2: for l in 0 .. 4, in order"], size=12, color="#4a5568")
            bar = "end of loop 1 is the barrier"
        d.box(px + PW / 2, Y_BAR, 380, 28, [bar], "result")
        cells = [d.box(px + OFF[l], Y_SH, BOX_W, 34, [f"{shared[l]:g}"], "plain") for l in range(n)]
        d.text(px + PW / 2, Y_SH + 36, ["shared array s, one per workgroup"], size=12, color="#4a5568")
        note = "zeroed when the workgroup starts" if p == 0 else "one stack array, cleared at the start of each workgroup"
        d.text(px + PW / 2, Y_SH + 54, [note], size=12, color="#4a5568")
        if p == 1:
            d.text(px + PW / 2, Y_SH + 72, ["mask alive[l] carries a return across the two loops"], size=12, color="#4a5568")
        _ = cells
    return d.svg()
