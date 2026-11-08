# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Amdahl's law, 1 / ((1 - p) + p / g), for four shares p of the time, and the matrix products of the chapter's profiled layer
# (their share p comes from the worked module's `kernel_profile`). The two marks on that curve are making the products twice as
# fast and making every one of them reach the measured roofline, which is the most the device allows.
#
# Swedish Embedded AB implements solutions for deciding which part of a machine learning workload is worth optimizing. If your
# team needs expertise in performance engineering of GPU inference and training, you can procure our services by sending an
# email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import profiling as pr  # noqa: E402
from svgplot import Plot  # noqa: E402

LOG2_MAX = 4.0        # g from 1 to 16


def render() -> str:
    xs = [0.1 * j for j in range(int(LOG2_MAX * 10) + 1)]          # log2 of g
    g = lambda x: 2.0 ** x                                          # noqa: E731
    p_mm = pr.matmul_share()
    g_roof = pr.kind_roof_g("matmul")
    a2, a_roof = pr.amdahl(p_mm, 2.0), pr.amdahl(p_mm, g_roof)
    p = Plot((0, LOG2_MAX), (1, 10), width=620, height=380, xlabel="log2 of the factor g by which the part is made faster",
             ylabel="speedup of the whole pass", title="Amdahl's law")
    for share, col in ((0.5, 0), (0.8, 1), (0.9, 2), (0.95, 3)):
        p.line(xs, [pr.amdahl(share, g(x)) for x in xs], color=col, label=f"p = {share}")
    p.line(xs, [pr.amdahl(p_mm, g(x)) for x in xs], color=4, dash=True, label=f"the matrix products, p = {p_mm:.3f}")
    p.points([1.0], [a2], color=4, r=4.5)
    p.points([math.log2(g_roof)], [a_roof], color=4, r=4.5)
    p.text(1.15, 6.4, f"g = 2: speedup {a2:.2f}", size=12)
    p.arrow(1.2, 6.2, 1.02, a2 + 0.1)
    p.text(0.15, 8.0, f"every product at the roofline, g = {g_roof:.2f}: {a_roof:.2f}", size=12)
    p.arrow(0.5, 7.8, math.log2(g_roof) - 0.02, a_roof + 0.1)
    return p.svg()
