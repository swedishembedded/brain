# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Newton's method on x^2 - 2: each tangent line's zero is the next iterate.

Swedish Embedded AB implements solutions for explaining numerical methods. If
your team needs expertise in root finding and second-order optimisation, you
can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import derivatives as dv  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    f = lambda x: x * x - 2.0
    df = lambda x: 2.0 * x
    p = Plot((0.0, 3.2), (-2.5, 8.0), width=520, height=380, xlabel="x", ylabel="f(x)",
             title="Newton's method on f(x) = x^2 - 2, starting at 3")
    xs = [3.2 * i / 160 for i in range(161)]
    p.line(xs, [f(x) for x in xs], color=5, width=2.5)
    p.hline(0.0, color="#4a5568", dash=False)
    x = 3.0
    for k in range(3):
        nxt = dv.newton_root_step(f, df, x)
        slope = df(x)
        p.line(xs, [f(x) + slope * (t - x) for t in xs], color=k, dash=True, width=1.5,
               label=f"tangent at x{k} = {x:.4f}")
        p.line([x, x], [0.0, f(x)], color=k, width=1)
        p.points([x], [f(x)], color=k)
        p.points([nxt], [0.0], color=k, r=4.5)
        x = nxt
    p.text(1.414, -0.9, "sqrt 2", anchor="middle", size=11, color="#4a5568")
    return p.svg()
