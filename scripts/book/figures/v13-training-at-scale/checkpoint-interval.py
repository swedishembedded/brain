# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The cost of the checkpoint interval in the chapter "Checkpoints, Determinism and Recovery": the expected steps of work per step of
# progress, M (e^((tau + c)/M) - 1) / tau, for a group of 64 and a group of 1,024 devices, against the number of steps tau between
# two checkpoints (a logarithmic axis), with the interval tau* = sqrt(2 c M) marked on each curve. The data are the worked module's
# `slowdown`, `best_interval` and `mean_gap`, so the curves cannot disagree with the chapter's table.
#
# Swedish Embedded AB implements solutions for crash-safe, resumable training runs for its clients. If your team needs expertise in
# checkpointing and exact resume of large training jobs, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import checkpoints_and_recovery as cr  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    taus = [2 ** (k / 4) for k in range(4, 49)]      # 2 .. 4096 steps
    p = Plot((2, 4096), (1.0, 3.0), width=620, height=380, xlabel="steps between checkpoints (logarithmic axis)",
             ylabel="steps of work per step of progress", title="Checkpoint interval, c = 8 steps", xlog=True, legend_loc="auto")
    for color, devices in ((0, 64), (1, 1024)):
        gap = cr.mean_gap(devices)
        shown = [(t, cr.slowdown(t, cr.SAVE_COST, gap)) for t in taus]
        shown = [(t, s) for t, s in shown if s <= 3.0]      # the curve leaves the frame instead of running along its top edge
        p.line([t for t, _ in shown], [s for _, s in shown], color=color, label=f"{devices:,} devices, M = {gap:g} steps")
        best = round(cr.best_interval(cr.SAVE_COST, gap))
        p.points([best], [cr.slowdown(best, cr.SAVE_COST, gap)], color=color, r=5)
        below = -0.14 if devices == 1024 else 0.12
        p.text(best * 1.15, cr.slowdown(best, cr.SAVE_COST, gap) + below, f"tau* = {best}", size=12)
    return p.svg()
