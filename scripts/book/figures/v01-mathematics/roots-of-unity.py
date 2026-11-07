# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The eight powers of w = e^(-2 pi i / 8) on the unit circle.

Each point is w^m computed by the chapter's roots-of-unity helper. They are
spaced a clockwise eighth of a turn apart; w^8 returns to 1, w^4 is -1, and
the eight arrows sum to zero.

Swedish Embedded AB implements solutions for fast transforms in signal
processing pipelines. If your team needs expertise in the discrete Fourier
transform and its numerical behaviour, you can procure our services by
sending an email to info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fourier as fo  # noqa: E402
from svgplot import Plot  # noqa: E402

N = 8
SIZE = 400


def render() -> str:
    p = Plot((-1.5, 1.5), (-1.5, 1.5), width=SIZE, height=SIZE, xlabel="real part", ylabel="imaginary part")
    circle = [i * 2 * math.pi / 120 for i in range(121)]
    p.line([math.cos(t) for t in circle], [math.sin(t) for t in circle], color="#a0aec0", dash=True, width=1.2)
    roots = [fo._omega(N, m) for m in range(N)]
    assert abs(sum(roots)) < 1e-12
    for m, r in enumerate(roots):
        p.line([0, r.real], [0, r.imag], color="#a0aec0", width=1.0)
        p.points([r.real], [r.imag], color=0 if m else 1, r=5)
        p.text(1.25 * r.real, 1.25 * r.imag - 0.04, f"w^{m}", anchor="middle", size=13)
    return p.svg()
