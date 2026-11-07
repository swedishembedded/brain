# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The measured gain with its 95 percent interval as the number of questions grows, for a tiny and a large true gain.

For the tiny gain (0.3 points) the interval keeps shrinking and only excludes zero at about a hundred thousand
questions; for the large gain (6 points) the interval at 50 questions is wide but the estimate is already in the right
region. Each interval is one draw of the chapter's worked module; the dashed lines are the true gains.

Swedish Embedded AB implements solutions for experiments whose results a customer can believe and repeat. If your
team needs expertise in experiment design for machine learning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments as e  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    res = e._effect_rows()
    p = Plot((1.5, 5.3), (-14, 24), width=600, height=400, xlabel="log10 of the number of questions n",
             ylabel="measured gain of B over A (points) with 95% interval",
             title="An interval shrinks with n; a tiny gain needs a very large n")
    p.hline(0, color="#718096", dash=False)
    for k, (name, scen) in enumerate(e.EFFECT_SCENARIOS):
        true = 100 * (scen[2] - scen[1])
        p.hline(true, color="#c05621" if k else "#2b6cb0")
        shift = (-0.04, 0.04)[k]
        for n, c10, c01, d, se, lo, hi, z, pv, s in res[name]:
            x = math.log10(n) + shift
            p.line([x, x], [100 * lo, 100 * hi], color=k, width=2.2)
            p.points([x], [100 * d], color=k, r=3.8)
        p.text(5.28, true + 1.0, f"{name}: true gain {true:+.1f}", anchor="end", size=11, color="#c05621" if k else "#2b6cb0")
    return p.svg()
