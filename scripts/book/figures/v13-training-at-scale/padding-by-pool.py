# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The share of computed positions that carry no record token, for 240 records in batches of 8, as a function of the size of the pool
# inside which records are sorted by length.
#
# Swedish Embedded AB implements solutions for cutting the padding out of training batches. If your team needs expertise in
# efficient fine-tuning data pipelines, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_data as td  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    useful = sum(td.record_lengths())
    pools = list(td.POOLS)
    pad = []
    for p in pools:
        _, computed, _ = td._batch_numbers(p)
        pad.append(100.0 * (1 - useful / computed))
    pl = Plot((1, 30), (0, 100), width=620, height=340, xlabel="pool size, in batches (log scale)",
              ylabel="padding, percent of positions computed", title="Padding falls as the pool grows", xlog=True, legend_loc=None)
    pl.line(pools, pad, color=0)
    pl.points(pools, pad, color=0)
    return pl.svg()
