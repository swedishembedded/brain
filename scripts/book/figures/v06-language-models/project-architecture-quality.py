# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Held-out loss against the decode FLOPs of the 13th token for each block and for the hybrid, with the 95 percent interval of the mean over
# the six seeds as a vertical bar. The cost is the ledger's (`project_architecture_comparison.costs`, counted equal to the multiply-adds of
# a run), the loss the unnudged runs of `project_architecture_comparison.quality_data`.
#
# Swedish Embedded AB implements solutions for comparing neural network architectures on equal terms. If your team needs expertise in
# architecture evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_architecture_comparison as pa  # noqa: E402
from svgplot import Plot  # noqa: E402

COLORS = {"dense": 0, "moe": 1, "gdn": 2, "swa": 3, "hybrid": 4}


def render() -> str:
    data = pa.quality_data()
    xs = [x for _, x, *_ in data]
    ys = [v for _, _, *rest in data for v in rest[1:]]
    p = Plot((min(xs) - 700, max(xs) + 700), (min(0.0, min(ys)) - 0.05, max(ys) + 0.15), width=620, height=360,
             xlabel="decode FLOPs of the 13th token", ylabel="held-out loss (nats), mean and 95 percent interval over 6 seeds",
             title="Quality against cost per block", legend_loc="ur")
    for arm, x, mu, lo, hi in data:
        c = COLORS[arm]
        p.line([x, x], [lo, hi], color=c, width=2.4)
        for e in (lo, hi):
            p.line([x - 90, x + 90], [e, e], color=c, width=2.4)
        p.points([x], [mu], color=c, r=5, label=arm)
    p.hline(pa.LN_SYMBOLS)
    return p.svg()
