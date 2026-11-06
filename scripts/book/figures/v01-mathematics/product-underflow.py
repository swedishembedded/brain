# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""A product of many small probabilities falls off the edge of the machine's numbers; the sum of logs does not.

Swedish Embedded AB implements solutions for numerically stable probability
computations in machine learning code. If your team needs expertise in
log-space arithmetic, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numbasics  # noqa: E402
from svgplot import Plot  # noqa: E402

FLOOR = -330.0  # where the plot draws "the machine stores exactly zero"


def render() -> str:
    ns = list(range(0, 401, 4))
    rows = numbasics.underflow_rows(ns)
    # log10 of the stored product; a stored zero is drawn on the floor of the plot.
    stored64 = [math.log10(r[1]) if r[1] > 0 else FLOOR for r in rows]
    stored32 = [math.log10(r[2]) if r[2] > 0 else FLOOR for r in rows]
    via_logs = [r[3] / math.log(10) for r in rows]  # log10 of the true product, from the sum of logs
    p = Plot((0, 400), (-420, 10), width=600, height=380, xlabel="number of factors n, each equal to 0.1",
             ylabel="log10 of the value (stored zero drawn at -330)", title="Multiplying 0.1 n times")
    p.line(ns, stored32, color=1, label="product, 32-bit numbers")
    p.line(ns, stored64, color=0, label="product, 64-bit numbers")
    p.line(ns, via_logs, color=2, dash=True, label="sum of logs (n * log10 0.1)")
    p.text(52, -360, "32-bit product is zero from n = 46", size=11, color="#c05621")
    p.text(52, -390, "64-bit product is zero from n = 324", size=11, color="#2b6cb0")
    return p.svg()
