# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The numbers a block stack holds per token at the peak of the backward pass, from the chapter "Where Training Memory Goes": the stack
# keeps the input of every k-th block and rebuilds a segment of k blocks before its backward, so the peak is (L / k) D + k (D + F). The data
# are the worked module's `segment_curve`, so the curve cannot disagree with the chapter's table.
#
# Swedish Embedded AB implements solutions for memory-bounded training of large models for its clients. If your team needs expertise in
# fitting a training run into a fixed device then you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_memory as tm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curve, keep = tm.segment_curve()
    ks = [k for k, _ in curve]
    p = Plot((1, tm.L), (5e4, 1e6), width=620, height=380, xlabel="segment length k (blocks rebuilt together)",
             ylabel="numbers held per token at the peak", title="Peak of the backward against the segment length",
             xlog=True, ylog=True, legend_loc="ll")
    p.line(ks, [v for _, v in curve], color=0, label="recompute, segments of k")
    p.hline(keep)
    p.text(1.1, keep * 1.12, f"keep every block: {keep:,}", size=12)
    divisors = [k for k in ks if tm.L % k == 0]
    p.points(divisors, [tm._seg_numbers(k) for k in divisors], color=1, label="k divides L")
    return p.svg()
