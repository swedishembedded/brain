# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The closed-loop concurrency ladder of the benchmarking chapter: output rate and goodput of the model engine against the
# number of requests in flight, both computed by the worked module's simulator (`benchmarking._sweep`). The marks are the
# peak output rate and the highest goodput among the levels at which the service-level objective held for 0.99 of the
# requests.
#
# Swedish Embedded AB implements solutions for benchmarking inference servers against the latency their users are
# promised. If your team needs expertise in serving performance and capacity planning, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import benchmarking as bm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    pts = bm._sweep()
    cs = [c for c, _ in pts]
    out = [s["tok_s"] for _, s in pts]
    good = [s["good_s"] for _, s in pts]
    held = [(c, s["good_s"]) for c, s in pts if s["attain"] >= 0.99]
    best = max(held, key=lambda p: p[1])
    peak = max(zip(cs, out), key=lambda p: p[1])
    p = Plot((1, 50), (0, 1300), width=620, height=380, xlog=True, xlabel="requests in flight (closed loop)",
             ylabel="tokens per second (model)", title="Peak output rate and goodput under the objective",
             legend_loc="ul")
    p.line(cs, out, color=0, label="output tokens per second")
    p.points(cs, out, color=0)
    p.line(cs, good, color=1, dash=True, label="goodput: tokens of requests that met the objective")
    p.points(cs, good, color=1)
    p.points([best[0]], [best[1]], color=2, r=6)
    p.text(1.1, 900, f"highest goodput with the objective held: {best[1]:,.0f} at {best[0]}", size=12)
    p.arrow(5.0, 880, best[0] * 0.95, best[1] + 25)
    p.text(peak[0], peak[1] + 60, f"peak output: {peak[1]:,.0f} at {peak[0]}", anchor="middle", size=12)
    return p.svg()
