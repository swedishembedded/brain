# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Four counters of a toy scheduler sampled at the end of every step, in the chapter "Running and Observing an Engine": the live
# queue depth, its never-resetting high-water mark, the jobs admitted onto the lane and the jobs counted at completion. The
# values are the worked module's `counter_series`, drawn as steps because a counter holds its value between snapshots.
#
# Swedish Embedded AB implements solutions for observing, controlling and cleanly stopping long-running inference servers for
# its clients. If your team needs expertise in serving metrics and their meaning then you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import operations as op  # noqa: E402
from svgplot import Plot  # noqa: E402


def _steps(xs, ys):
    """A step line: each value holds from its step to the next."""
    sx, sy = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i:
            sx.append(x)
            sy.append(ys[i - 1])
        sx.append(x)
        sy.append(y)
    sx.append(xs[-1] + 1)
    sy.append(ys[-1])
    return sx, sy


def render() -> str:
    run = op.counter_series()
    xs = [r["step"] for r in run]
    p = Plot((0, 16), (0, 8), width=620, height=360, xlabel="step (snapshot at its end)", ylabel="jobs",
             title="Four counters of one lane, batches of at most 2", legend_loc="ul")
    for color, (key, label, dash) in enumerate((("admitted", "admitted", False), ("jobs", "jobs (at completion)", False),
                                               ("queue_peak", "queue_peak", True), ("queue_depth", "queue_depth", False))):
        sx, sy = _steps(xs, [r[key] for r in run])
        p.line(sx, sy, color=color, label=label, dash=dash)
    return p.svg()
