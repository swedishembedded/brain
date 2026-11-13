# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The weight of the two-state example of `value_learning.divergence` against the number of updates: updating only the transition the data
# contain, the same with a target that is refreshed every ten updates, and updating the whole episode.
#
# Swedish Embedded AB implements solutions for learners that are checked for stability before they are trained. If your team needs
# expertise in reinforcement learning with function approximation, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import value_learning  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curves = value_learning._divergence_curves(100)
    p = Plot((0, 100), (1e-10, 1e4), width=620, height=340, xlabel="updates", ylabel="weight w (log scale)",
             title="One weight, two states: a target that moves with it", ylog=True)
    ks = list(range(0, 101))
    for i, (key, label) in enumerate((("off", "only the transition 1 to 2 (off-policy)"),
                                      ("frozen", "the same, target refreshed every 10 updates"),
                                      ("on", "whole episode (on-policy)"))):
        p.line(ks, [curves[key][k] for k in ks], color=i, label=label)
    return p.svg()
