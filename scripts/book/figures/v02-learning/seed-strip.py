# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Test accuracy of two pipelines over 100 seeds: one dot per seed, a bar at each mean.

Both pipelines see the same data and the same test rows; only the seed (initial weights and data order) changes
between dots. The plain pipeline ranges over thirteen points, the other over two, so a single seed of either one is
a poor summary, and a comparison of one dot against one dot can land anywhere. The numbers come from the chapter's
worked module.

Swedish Embedded AB implements solutions for experiments whose results a customer can believe and repeat. If your
team needs expertise in experiment design for machine learning, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments as e  # noqa: E402
from svgplot import Plot  # noqa: E402


def _arm_labels(svg: str, labels: dict) -> str:
    """Replace the numeric x tick labels by names (and blank the ticks that name nothing)."""
    def repl(m):
        return m.group(1) + labels.get(m.group(2), "") + "</text>"
    return re.sub(r'(font-size="11" fill="#4a5568" text-anchor="middle">)([0-9.]+)</text>', repl, svg)


def render() -> str:
    arms = (("plain", "plain pipeline"), ("both", "centring + interaction"))
    p = Plot((0.4, 2.6), (60, 77), width=560, height=380, ylabel="test accuracy (%)",
             title=f"Test accuracy over {e.N_RUNS} seeds, same data, same test rows")
    for i, (key, _) in enumerate(arms):
        acc = [100 * a for a in e._seed_accuracies(key)]
        xs = [i + 1 + ((k * 0.6180339887) % 1.0 - 0.5) * 0.55 for k in range(len(acc))]
        p.points(xs, acc, color=i, r=3.0)
        m = sum(acc) / len(acc)
        p.line([i + 1 - 0.36, i + 1 + 0.36], [m, m], color="#1a202c", width=2.4)
        sd = e._sd(acc)
        p.text(i + 1, 62, f"mean {m:.1f}, sd {sd:.1f}", anchor="middle", size=12)
    svg = p.svg()
    return _arm_labels(svg, {"1": arms[0][1], "2": arms[1][1]})
