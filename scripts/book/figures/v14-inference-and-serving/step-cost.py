# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The cost in equivalent bytes of a decode step of 16 sequences at context 2,048 that also carries n prompt rows, in one
# forward and as two forwards, against n, in the chapter "Batching and Scheduling Requests". The costs are the worked module's
# `forward_cost50`, so the curves cannot disagree with the chapter's table.
#
# Swedish Embedded AB implements solutions for serving language models to many users at once on the hardware they already
# have. If your team needs expertise in chunked prefill and mixed batches then you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import batching as bt  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = list(range(0, 513, 4))
    one = bt.cost_curve(16, 2048, rows)
    two = bt.cost_curve(16, 2048, rows, separate=True)
    knee = bt.free_rows(16, 2048)
    p = Plot((0, 512), (0, 180), width=640, height=380, xlabel="prompt rows carried with the decode step",
             ylabel="cost (10^9 equivalent bytes)", title="A decode step of 16 sequences at context 2,048 plus prompt rows",
             legend_loc="ul")
    p.line([n for n, _ in two], [c for _, c in two], color=1, label="prompt rows in a forward of their own", dash=True)
    p.line([n for n, _ in one], [c for _, c in one], color=0, label="prompt rows in the decode forward")
    p.vline(knee)
    p.text(knee + 8, 8, f"{knee} rows: the decode forward leaves the bandwidth bound", size=12, plate=True)
    return p.svg()
