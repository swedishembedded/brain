# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The factor by which each scheme divides the rotation rate of each pair.

Head width 64, base 10,000, trained length 4,096, extension 8 (`long_context.rate_ratio_data`): position interpolation divides every
pair by 8, NTK-aware scaling by 8^(2m/62), YaRN keeps the fast pairs and interpolates the slow ones.

Swedish Embedded AB implements solutions for serving language models beyond the length they were trained on. If your team needs
expertise in rotary position scaling, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import long_context as lc  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    r = lc.rate_ratio_data()
    m = list(range(lc.D_HEAD // 2))
    p = Plot((0, 31), (0.1, 1.2), width=620, height=380, xlabel="frequency pair m (fast to slow)",
             ylabel="scaled rate / original rate", title="How each scheme rescales the rates (s = 8)", ylog=True, legend_loc="ll")
    p.line(m, r["interpolation"], color=0, label="position interpolation")
    p.line(m, r["NTK-aware"], color=1, label="NTK-aware")
    p.line(m, r["YaRN"], color=2, label="YaRN, ramp over the pair index")
    p.line(m, r["YaRN, ramp in turns"], color=4, dash=True, label="YaRN, ramp over the turns")
    return p.svg()
