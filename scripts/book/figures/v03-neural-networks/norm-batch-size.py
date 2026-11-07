# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Train-mode batch-norm output of one probe example against the batch size (computed by worked module
`normalization`, function `batch_size_summary`): its mean and the 5th and 95th percentiles over 1000 batches whose other
members are random. The eval-mode output (exact population statistics) is the dashed line at 1.

Swedish Embedded AB implements solutions for stable training of networks with small batches. If your team needs expertise
in batch normalization and its alternatives, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import normalization as nm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = nm.batch_size_summary()
    xs = [math.log2(b) for b, *_ in rows]
    p = Plot((0, 7), (-1.5, 2.0), width=620, height=360, xlabel="log2 of the batch size B (B = 1, 2, 4, ..., 128)",
             ylabel="output for the probe value", title="Batch norm in train mode depends on the batch")
    p.line([0, 7], [1.0, 1.0], color=5, width=1.5, dash=True, label="eval mode, exact statistics: 1")
    p.line(xs, [r[3] for r in rows], color=1, width=2.0, label="5th percentile")
    p.line(xs, [r[4] for r in rows], color=2, width=2.0, label="95th percentile")
    p.line(xs, [r[1] for r in rows], color=0, width=2.5, label="mean over 1000 batches")
    p.points(xs, [r[1] for r in rows], color=0)
    return p.svg()
