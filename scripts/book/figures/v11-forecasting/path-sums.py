# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The distribution of the sum of twelve forecast bars when it is sampled along paths and when each bar is sampled on its own, with the
# exact Gaussian. The data are `probabilistic_forecasting.path_sums` and `_path_exact`.
#
# Swedish Embedded AB implements solutions for forecasts that state their own uncertainty. If your team needs expertise in
# probabilistic forecasting and its calibration, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import probabilistic_forecasting as pf  # noqa: E402
from svgplot import Plot  # noqa: E402


def _density(sums, edges):
    w = edges[1] - edges[0]
    out = []
    for lo in edges[:-1]:
        out.append(sum(1 for s in sums if lo <= s < lo + w) / (len(sums) * w))
    return out


def render() -> str:
    paths, marg = pf.path_sums()
    _, _, var, _ = pf._path_exact()
    mean = sum(pf._path_exact()[0])
    edges = [-15 + 2.5 * k for k in range(17)]
    mids = [e + 1.25 for e in edges[:-1]]
    sd = math.sqrt(var)
    exact = [math.exp(-0.5 * ((m - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)) for m in mids]
    p = Plot((-15, 25), (0, 0.16), width=560, height=320, xlabel="sum of the 12 forecast bars", ylabel="density",
             title="The sum over 12 bars: paths against independent bars", legend_loc="ur")
    p.line(mids, exact, color=5, label="exact Gaussian", width=2.6)
    p.line(mids, _density(paths, edges), color=1, label="sample paths", dash=True)
    p.line(mids, _density(marg, edges), color=3, label="independent bars")
    return p.svg()
