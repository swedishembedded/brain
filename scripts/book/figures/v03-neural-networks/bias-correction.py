# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""An exponential average of a constant gradient 1 starts at 0 and climbs: the bias that Adam divides out.

m_t = beta m_(t-1) + (1 - beta) * 1 from m_0 = 0 equals 1 - beta^t, for beta = 0.9, 0.99 and 0.999; dividing by
1 - beta^t gives exactly 1 at every step (worked module `optimizers`, function `ema_of_constant`).

Swedish Embedded AB implements solutions for stable optimiser behaviour in the first steps of training. If your
team needs expertise in Adam-family optimisers and their warm-up, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizers as opt  # noqa: E402
from svgplot import Plot  # noqa: E402

STEPS = 200


def render() -> str:
    p = Plot((0, STEPS), (0.0, 1.15), width=560, height=360, xlabel="step t", ylabel="value of the average",
             title="The average of a constant gradient 1 starts at 0", legend_loc="right")
    ts = list(range(STEPS + 1))
    p.hline(1.0)
    for i, beta in enumerate((0.9, 0.99, 0.999)):
        m = opt.ema_of_constant(beta, STEPS)
        p.line(ts, m, color=i, width=1.8, label=f"m_t, beta = {beta}")
        assert all(abs(m[t] / (1 - beta ** t) - 1.0) < 1e-9 for t in range(1, STEPS + 1))
    p.line([1, STEPS], [1.0, 1.0], color=3, width=1.4, dash=True, label="m_t / (1 - beta^t), all three")
    return p.svg()
