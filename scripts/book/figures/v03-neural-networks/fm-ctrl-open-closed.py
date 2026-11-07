# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Open-loop and closed-loop errors of the controller during training, per epoch, on a base-10 logarithmic scale.

From the log of `first_models.ctrl_open_vs_closed`: the RMS command error on validation samples (one step, every
input an expert state), the same error at the states the network visits in its own closed loop, and the steady
tracking error at the end of the holds.

Swedish Embedded AB implements solutions for learned and classical control of embedded systems. If your team
needs expertise in learned controllers, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import first_models as fm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    log = fm.ctrl_model("explore")["log"]
    lg = math.log10
    ep = [r["epoch"] for r in log]
    p = Plot((1, fm.CTRL_RUN["epochs"]), (-2.5, 0.5), width=620, height=360, xlabel="epoch", ylabel="log10 of the error",
             title="Controller: open-loop and closed-loop errors while training")
    p.line(ep, [lg(r["open_rms"]) for r in log], color=0, label="command error, one step (validation)")
    p.line(ep, [lg(r["closed_rms"]) for r in log], color=1, label="command error, own states")
    p.line(ep, [lg(r["ss"]) for r in log], color=2, label="steady tracking error")
    return p.svg()
