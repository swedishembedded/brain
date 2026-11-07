# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Running averages of the two-dice sum converging to its expectation, 7.

Three independent runs, each from its own bookrng generator (names probability.running_average_figure.0 to .2), are averaged
with the chapter's own `running_average` helper; the dashed line is the
expectation computed from the probability table.

Swedish Embedded AB implements solutions for statistical evaluation of models.
If your team needs expertise in Monte Carlo estimation and its error, you can
procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probability as pr  # noqa: E402
from svgplot import Plot  # noqa: E402

N = 2000


def render() -> str:
    expectation = float(pr._expect(pr._sum_pmf()))
    p = Plot((0, N), (4.5, 9.5), width=560, height=340, xlabel="number of rolls n",
             ylabel="average of the sum so far", title="Averages settle on the expectation")
    xs = list(range(1, N + 1))
    for k in range(3):
        avg = pr.running_average(N, f"probability.running_average_figure.{k}")
        p.line(xs, avg, color=k, label=f"run {k + 1}", width=1.6)
    p.hline(expectation, color="#2d3748")
    p.text(N * 0.98, expectation + 0.5, f"E[sum] = {expectation:g}", anchor="end", size=12, color="#2d3748")
    return p.svg()
