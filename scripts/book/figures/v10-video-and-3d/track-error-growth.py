# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The mean position error of the learned linear cart model after k free-running steps (`world_models.track_fit`), against its
# one-step error when it is fed the true state at every step.
#
# Swedish Embedded AB implements solutions for learned simulators whose rollout error is measured, not assumed. If your team needs
# expertise in world models and long-horizon prediction, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import world_models as wm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    err, teacher = wm._error_curve()
    ks = list(range(1, len(err)))
    p = Plot((1, ks[-1]), (0, 2.6), width=560, height=320, xlabel="free-running steps k", ylabel="mean |position error|",
             title="Error of the learned cart model")
    p.hline(teacher)
    p.text(ks[-1], teacher + 0.08, "one step, true state fed back", anchor="end", size=11)
    p.line(ks, [err[k] for k in ks], color=0, width=2.0, label="model fed its own output")
    return p.svg()
