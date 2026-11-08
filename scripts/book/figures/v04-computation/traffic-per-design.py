# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Bytes read, bytes written and their total (MiB) of the chain y = layer_norm(x + gelu(x W + b)) on [2048, 1024], against the
# number of dispatches it is run in (five, two, one), with the floor every design is bounded by (each input read once, the
# output written once). The data are the worked module's `fusion_levels` counts.
#
# Swedish Embedded AB implements solutions for removing the memory round trips between the small kernels of a machine learning
# workload. If your team needs expertise in memory-traffic analysis for GPU inference and training, you can procure our
# services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fusion as fu  # noqa: E402
from svgplot import Plot  # noqa: E402

MIB = 1 << 20


def render() -> str:
    levels, _, _ = fu.figure_numbers()
    xs = [n for _, n, _, _ in levels]
    reads = [r / MIB for _, _, r, _ in levels]
    writes = [w / MIB for _, _, _, w in levels]
    totals = [(r + w) / MIB for _, _, r, w in levels]
    floor = totals[-1]
    p = Plot((0, 6), (0, 110), width=600, height=360, xlabel="dispatches the chain runs in",
             ylabel="MiB through memory", title="Memory traffic of the chain against fusion")
    p.line([0.4, 5.6], [floor, floor], color=5, dash=True, label="floor: each input once, y once")
    p.line(xs, totals, color=0, label="total")
    p.points(xs, totals, color=0)
    p.line(xs, reads, color=1, label="read")
    p.points(xs, reads, color=1)
    p.line(xs, writes, color=2, label="written")
    p.points(xs, writes, color=2)
    return p.svg()
