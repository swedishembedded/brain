# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Occupancy of one compute unit against the registers each invocation uses, for workgroups of 64, 256 and 1,024
# invocations, with no shared memory. Every point is the worked module's `occupancy` model (65,536 registers,
# 2,048 resident invocations, 16 resident workgroups per compute unit). The dashed line is the occupancy that
# keeps 160,000 bytes in flight with one 4-byte load per invocation (500 of 2,048 invocations per compute unit).
#
# Swedish Embedded AB implements solutions for tuning the resource use of GPU kernels. If your team needs
# expertise in GPU kernel occupancy and register pressure, you can procure our services by sending an email to
# info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gpu  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    regs = list(range(16, 256))
    p = Plot((0, 256), (0, 108), width=620, height=380, xlabel="registers per invocation",
             ylabel="occupancy (% of 2,048 resident invocations)", title="Occupancy against registers")
    for color, wg in enumerate((64, 256, 1024)):
        p.line(regs, [100 * gpu.occupancy(wg, r, 0)[1] for r in regs], color=color, label=f"workgroup of {wg}")
    need = math.ceil(gpu.BW * gpu.MEM_LATENCY_CYCLES / gpu.CLOCK_HZ / gpu.F32 / gpu.N_CU)
    p.hline(100 * need / gpu.MAX_THREADS_PER_CU)
    p.text(252, 100 * need / gpu.MAX_THREADS_PER_CU + 3, f"needed to hide latency: {need} invocations", anchor="end",
           size=11)
    return p.svg()
