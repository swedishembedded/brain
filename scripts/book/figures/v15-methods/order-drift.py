# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The batch-size ladder of the benchmarking chapter on a device that slows as it heats: the steady (true) rate of each
# batch size against what four measurement orders report, every value from the worked module (`benchmarking._drift_orders`).
#
# Swedish Embedded AB implements solutions for performance measurements that separate a real effect from the drift of
# the machine. If your team needs expertise in benchmark design on shared or thermally limited hardware, you can procure
# our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmarking as bm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    o = bm._drift_orders()
    bs = list(bm.DRIFT_B)
    p = Plot((1, 8), (0.8, 1.3), width=620, height=380, xlog=False, xlabel="batch size B",
             ylabel="rate relative to B = 1", title="One ladder, three orders of measurement", legend_loc="ul")
    p.line(bs, list(bm.DRIFT_TRUE), color=4, width=3.0, label="steady rate (the truth)")
    p.line(bs, o["descending"], color=1, label="descending from cold")
    p.points(bs, o["descending"], color=1)
    p.line(bs, o["rest"], color=2, dash=True, label="cooled before each level")
    p.points(bs, o["rest"], color=2)
    p.line(bs, o["ascending"], color=0, label="ascending from cold")
    p.points(bs, o["ascending"], color=0)
    p.hline(1.0)
    return p.svg()
