# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The share of a 64 GiB cache pool's allocated token slots that hold a row, against the block size, for the 256 drawn
# conversations of the chapter "Managing the KV Cache". The data are the worked module's `paging_curve`, so the curve
# cannot disagree with the chapter's table; a block of 32,768 tokens is the per-conversation reservation.
#
# Swedish Embedded AB implements solutions for serving many concurrent conversations from the memory of one device for its
# clients. If your team needs expertise in KV cache paging then you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paged_kv as pk  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = pk.paging_curve([2 ** i for i in range(16)])
    xs = [b for b, _, _ in data]
    p = Plot((1, 32768), (0, 105), width=620, height=380, xlog=True, xlabel="block size (tokens)",
             ylabel="allocated slots holding a row (%)", title="How much of the pool the blocks fill", legend_loc="ll")
    p.line(xs, [100.0 * u for _, _, u in data], color=0, label="share of allocated slots holding a row")
    p.points(xs, [100.0 * u for _, _, u in data], color=0)
    return p.svg()
