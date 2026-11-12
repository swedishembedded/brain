# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The probability integral transform of the test targets under the Gaussian and the Student-t head, in ten bins, against the count
# a calibrated head would give. The data are `probabilistic_forecasting.pit_counts`.
#
# Swedish Embedded AB implements solutions for forecasts that state their own uncertainty. If your team needs expertise in
# probabilistic forecasting and its calibration, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probabilistic_forecasting as pf  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    counts = pf.pit_counts()
    mid = [(j + 0.5) / pf.PIT_BINS for j in range(pf.PIT_BINS)]
    n = pf.TEST_A_ROWS[1] - pf.TEST_A_ROWS[0] + 1
    p = Plot((0, 1), (0, 160), width=560, height=320, xlabel="u = F(y), the head's distribution function at the target",
             ylabel="test rows in the bin", title="Where the targets fall in each head's distribution", legend_loc="ul")
    for i, (label, c) in enumerate(counts.items()):
        p.line(mid, c, color=i, label=label)
        p.points(mid, c, color=i, r=3)
    p.hline(n / pf.PIT_BINS)
    return p.svg()
