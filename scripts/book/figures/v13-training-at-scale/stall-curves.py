# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The distance of one weight from its target under four ways of storing it, from the chapter "Mixed-Precision Training": the descent of
# (w - t)^2 / 2 with fp32 storage, bf16 storage rounded to nearest, a bf16 working copy of an fp32 master, and bf16 storage rounded
# stochastically (run 0). The data are the worked module's `_stall_runs`, so the curves cannot disagree with the chapter's table.
#
# Swedish Embedded AB implements solutions for memory-bounded low-precision training for its clients. If your team needs expertise in
# training large models in 16 and 8 bits, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mixed_precision as mp  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    runs = mp._stall_runs()
    t = mp.STALL_TARGET
    steps = list(range(1, mp.STALL_STEPS + 1))
    p = Plot((1, mp.STALL_STEPS), (1e-7, 4.0), width=620, height=380, xlabel="step", ylabel="distance |w - t| of the stored weight",
             title="One weight, four storage rules", xlog=True, ylog=True, legend_loc="ll")
    err = lambda tr: [max(abs(w - t), 1e-7) for w in tr]  # noqa: E731
    p.line(steps, err(runs["f32"][0]), color=0, label="fp32 weight")
    p.line(steps, err(runs["rne"][0]), color=1, label="bf16 weight, nearest even")
    p.line(steps, err(runs["master"][0]), color=2, label="bf16 working copy of an fp32 master")
    p.line(steps, err(runs["sr"][0]), color=3, label="bf16 weight, stochastic rounding (run 0)")
    radius = mp.gap_bf16(3.0) / (2 * mp.STALL_LR)
    p.hline(radius)
    p.text(1.3, radius * 1.25, f"nearest even stops inside {radius:.4f}", size=12)
    return p.svg()
