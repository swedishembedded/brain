# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The distribution over eight bins that holds one side of a box, for a true distance of 3.25 cells: the two-hot optimum of the
# distribution focal loss, a wide bump and a spike on bin 3 (`detection.dfl_bars_data`).
#
# Swedish Embedded AB implements solutions for box regression heads whose outputs are distributions rather than single numbers. If your
# team needs expertise in detection heads and their losses, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import detection as dt  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    (q_opt, e_opt, l_opt), (q_wide, e_wide, l_wide), (q_spike, e_spike, l_spike) = dt.dfl_bars_data()
    p = Plot((-0.5, dt.NBIN - 0.5), (0, 1.05), width=560, height=380, xlabel="bin k (distance in cells)", ylabel="probability q_k",
             title="One box side as a distribution over bins", legend_loc="below")
    series = [(q_opt, 0, f"two-hot optimum: E = {e_opt:.2f}, DFL {l_opt:.2f}"),
              (q_wide, 1, f"wide bump: E = {e_wide:.2f}, DFL {l_wide:.2f}"),
              (q_spike, 2, f"spike on bin 3: E = {e_spike:.2f}, DFL {l_spike:.2f}")]
    xs = list(range(dt.NBIN))
    for q, color, label in series:
        p.line(xs, q, color=color, width=2.0)
        p.points(xs, q, color=color, label=label, r=3.8)
    p.vline(dt.TARGET_D, color="#718096")
    p.text(dt.TARGET_D + 0.12, 1.0, "true distance 3.25", size=12)
    return p.svg()
