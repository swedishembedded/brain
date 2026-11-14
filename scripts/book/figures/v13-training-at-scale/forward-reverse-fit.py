# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A two-peaked teacher and the two best one-peaked students, one for each direction of the divergence, from the chapter
# "Distillation". The data are the worked module's `fits`, so the picture cannot disagree with the table.
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
    q, bf, br = distillation.fits()
    xs = list(range(distillation.POINTS))
    p = Plot((0, 9), (0, 0.6), width=620, height=380, xlabel="token", ylabel="probability",
             title="One peak cannot match two", legend_loc="ur")
    p.line(xs, q, color=5, label="teacher q")
    p.line(xs, bf[3], color=0, label="student, forward KL(q || p)")
    p.line(xs, br[3], color=3, label="student, reverse KL(p || q)")
    p.points(xs, q, color=5)
    p.points(xs, bf[3], color=0)
    p.points(xs, br[3], color=3)
    return p.svg()
