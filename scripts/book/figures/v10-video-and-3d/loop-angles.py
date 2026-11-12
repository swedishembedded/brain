# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The joint angle of the toy body (`neural_simulation.closed_loop`) with its nervous system intact and with the proprioceptive channel
# cut, over the 3 seconds of 1500 control ticks.
#
# Swedish Embedded AB implements solutions for closed-loop neural control of simulated bodies. If your team needs expertise in
# sensorimotor loops and their controls, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import neural_simulation as ns  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    a = ns._loop_angles()
    t = [(i + 1) * 0.002 for i in range(len(a["intact"]))]
    idx = list(range(0, len(t), 5))
    p = Plot((0, 3.0), (0, 0.35), width=560, height=320, xlabel="time (seconds)", ylabel="joint angle (radians)",
             title="The sensor closes the loop")
    p.line([t[i] for i in idx], [a["cut"][i] for i in idx], color=2, width=2.0, label="proprioception cut")
    p.line([t[i] for i in idx], [a["intact"][i] for i in idx], color=0, width=2.0, label="loop intact")
    return p.svg()
