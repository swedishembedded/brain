# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The test divergence of a student against the number of training contexts, for three kinds of target, from the chapter
# "Distillation". The data are the worked module's `curve` and `_floor`, so the picture cannot disagree with the table.
#
# Swedish Embedded AB implements solutions for compressing large models into small ones for its clients. If your team needs
# expertise in knowledge distillation then you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import distillation  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = distillation.curve()
    ns = [r[0] for r in rows]
    p = Plot((5, 55), (0.3, 1.4), width=620, height=380, xlabel="training contexts (log scale)", ylabel="mean KL(q || p), test contexts (log scale)",
             title="What each kind of target buys the student", xlog=True, ylog=True, legend_loc="ur")
    p.line(ns, [r[1] for r in rows], color=3, label="one sampled label")
    p.line(ns, [r[3] for r in rows], color=0, label="teacher's top 2")
    p.line(ns, [r[2] for r in rows], color=2, label="teacher's full distribution")
    p.points(ns, [r[1] for r in rows], color=3)
    p.points(ns, [r[3] for r in rows], color=0)
    p.points(ns, [r[2] for r in rows], color=2)
    p.hline(distillation._floor())
    p.text(5.4, distillation._floor() * 0.93, "400 contexts, full distribution", size=12)
    return p.svg()
