# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Which of the 85 windows of the sample series are training windows and which are holdout windows under four ways of splitting them.
# The data are `time_series.policy_data`, the table of `window_policies`.
#
# Swedish Embedded AB implements solutions for splits of time-series data that cannot leak the future. If your team needs expertise
# in leakage-safe evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time_series as ts  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    pol = ts.policy_data()
    o_all = ts._origins()
    p = Plot((ts.CONTEXT - 2, ts.N - ts.WINDOW_H + 2), (0.4, 4.9), width=620, height=330, xlabel="origin o of the window (index of its first forecast bar)",
             ylabel="row of the split table", title="Windows by split", legend_loc="below")
    first = True
    names = ("at random", "a cut, no purge", "purge and embargo, gap 12", "purge and embargo, gap 16")
    for row, (name, tr, ho) in enumerate(pol, start=1):
        un = [o for o in o_all if o not in tr and o not in ho]
        p.points(un, [row] * len(un), color=5, r=2.2, label="unused" if first else "")
        p.points(tr, [row] * len(tr), color=0, r=2.6, label="training window" if first else "")
        p.points(ho, [row] * len(ho), color=1, r=2.6, label="holdout window" if first else "")
        first = False
        p.text(ts.CONTEXT - 1, row + 0.3, names[row - 1], size=11)
    p.vline(ts._cuts(ts.N, ts.TRAIN_FRAC, ts.VAL_FRAC)[0] + 0.5)
    return p.svg()
