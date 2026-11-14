# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Bytes that the busiest rank sends in an all-reduce of the 16 GiB gradient of the chapter's stack, in a coordinator star and in a
# ring, against the number of ranks. The data are the worked module's `wire_curve`.
#
# Swedish Embedded AB implements solutions for spreading training over several devices for its clients. If your team needs
# expertise in collective communication you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import parallelism  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curve = parallelism.wire_curve()
    rs = [r for r, _, _ in curve]
    p = Plot((2, 32), (0, 520), width=620, height=380, xlabel="ranks R", ylabel="GiB sent by the busiest rank",
             title="One all-reduce of a 16 GiB gradient", legend_loc="ul")
    p.line(rs, [b for _, b, _ in curve], color=0, label="coordinator star, rank 0")
    p.line(rs, [r for _, _, r in curve], color=1, label="ring, every rank")
    return p.svg()
