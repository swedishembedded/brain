# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Closed-loop response of the learned controller against the PID reference on the nominal plant.

The 180-step test of `first_models.ctrl_closed_loop`: setpoint 0.70, -0.60, 0.40, -0.85 held for 45 steps each. The
networks are the two checkpoints of that table, the reference the expert PI controller they imitate.

Swedish Embedded AB implements solutions for learned and classical control of embedded systems. If your team
needs expertise in learned controllers, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import first_models as fm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    pid = fm.rollout(fm.pid_controller())
    net = fm.rollout(fm.net_controller(fm.CTRL_SPEC, fm.ctrl_model("explore")["best"]))
    p = Plot((0, fm.ROLLOUT_STEPS), (-1.0, 1.0), width=640, height=360, xlabel="step (0.05 time units)", ylabel="y",
             title="Step responses on the nominal plant")
    p.line([r[0] for r in pid], [r[1] for r in pid], color=5, dash=True, label="setpoint")
    p.line([r[0] for r in pid], [r[2] for r in pid], color=0, label="PID reference")
    cloned = fm.rollout(fm.net_controller(fm.CTRL_SPEC, fm.ctrl_model("expert")["best"]))
    p.line([r[0] for r in cloned], [r[2] for r in cloned], color=2, label="learned, expert states only")
    p.line([r[0] for r in net], [r[2] for r in net], color=1, label="learned, explored states")
    return p.svg()
