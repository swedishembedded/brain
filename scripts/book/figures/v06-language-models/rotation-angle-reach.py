# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The largest rotation angle each frequency pair reaches in the trained window and in the extended one.

Head width 64, base 10,000, trained on positions 0 to 4,095 and read up to 32,767 (`long_context.angle_reach_data`). The dashed line
is one turn, 2 pi: a pair whose trained angle is below it has seen only part of its circle.

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
    data = lc.angle_reach_data()
    m = list(range(len(data)))
    p = Plot((0, 31), (0.05, 40000), width=620, height=380, xlabel="frequency pair m (fast to slow)",
             ylabel="largest angle (radians)", title="Angles reached at 4,095 and at 32,767", ylog=True, legend_loc="ll")
    p.line(m, [a for a, _ in data], color=0, label="trained window, position 4,095")
    p.line(m, [b for _, b in data], color=1, label="extended window, position 32,767")
    p.hline(lc.TAU, color="#a0aec0")
    p.text(0.5, lc.TAU * 1.25, "one turn = 6.28", size=11)
    never = [i for i in m if data[i][0] < lc.TAU]
    p.points(never, [data[i][0] for i in never], color=3, label="pairs that never completed a turn in training", r=3.5)
    return p.svg()
