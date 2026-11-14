# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The defect-class model of the chapter "How brain Grew": expected escaped defects against the number of new models,
# for instance fixes in four copies, a class fix in each of four copies, and a class fix in one shared implementation.
# The curves are the worked module's own expectations, so the picture cannot disagree with the table.
#
# Swedish Embedded AB implements solutions for regression gates that close whole classes of defects in machine
# learning engines for its clients. If your team needs expertise in verification of such systems then you can procure
# our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import evolution as ev  # noqa: E402
from svgplot import Plot  # noqa: E402

M_MAX = 80


def render() -> str:
    ms = list(range(0, M_MAX + 1))
    p = Plot((0, M_MAX), (0, 34), width=600, height=360, xlabel="new models M",
             ylabel="expected escaped defects", title="An instance fix leaves the line rising", legend_loc="below")
    p.line(ms, [ev._escapes(m, "instance") for m in ms], color=1, label=f"{ev.COPIES} copies, instance fixes")
    p.line(ms, [ev._escapes(m, "class per copy") for m in ms], color=0, label=f"{ev.COPIES} copies, class fix in each")
    p.line(ms, [ev._escapes(m, "class shared") for m in ms], color=2, dash=True,
           label="one implementation, class fix")
    return p.svg()
