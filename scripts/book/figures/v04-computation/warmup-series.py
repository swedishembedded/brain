# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The first 60 runs of the seeded warm-up series of the chapter's worked module (`warmup_runs`): the compile cost of run 0 (off
# the scale), the decaying clock ramp, the noise of Chapter 10, and the run at which the detection rule declares the warm-up over.
#
# Swedish Embedded AB implements solutions for benchmarks that do not time their own warm-up. If your team needs expertise in
# performance measurement of GPU workloads, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import profiling as pr  # noqa: E402
from svgplot import Plot  # noqa: E402

SHOWN = 60
TOP = 3.0


def render() -> str:
    t = pr._warm_series()
    end, _ = pr._warm_end(t)
    steady = sum(t[end:]) / len(t[end:])
    xs = list(range(SHOWN))
    p = Plot((0, SHOWN - 1), (0.9, TOP), width=620, height=360, xlabel="run number i", ylabel="time of run i (steady floors)",
             title="A seeded series: the warm-up and the noise")
    p.line(xs, [1.0 + pr.WARM_A * math.exp(-i / pr.WARM_TAU) for i in xs], color=2, dash=True, label="noise-free transient")
    p.line([0, SHOWN - 1], [steady, steady], color=3, label=f"mean after the warm-up, {steady:.3f}")
    p.points(xs[1:], [min(v, TOP) for v in t[1:SHOWN]], color=0, label="one run", r=3.0)
    p.vline(end)
    p.text(end + 0.8, TOP - 0.1, f"warm-up ends at run {end}", size=12)
    p.text(1.2, TOP - 0.1, f"run 0: {t[0]:.1f}, off the scale", size=12)
    return p.svg()
