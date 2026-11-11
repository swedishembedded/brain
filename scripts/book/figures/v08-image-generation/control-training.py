# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The loss of the toy denoiser's control branch over 100 steps of gradient descent, with its zero-convolutions starting at 0 and
# drawn like any other layer (`control_and_identity.loss_curves_data`, the first run of each), against the loss of the base network.
#
# Swedish Embedded AB implements solutions for steering image generators with a pose, an edge map or a face, checked by training a
# small model first. If your team needs expertise in control adapters, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import control_and_identity  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    steps, zero, drawn, base = control_and_identity.loss_curves_data()
    p = Plot((0, 100), (0, 0.32), width=560, height=340, xlabel="gradient step", ylabel="loss on the training samples",
             title="Zero-convolutions at 0 against drawn", legend_loc="ur")
    p.hline(base, color="#718096")
    p.line(steps, zero, color=0, label="zero-convolutions start at 0", width=2.2)
    p.line(steps, drawn, color=1, label="zero-convolutions drawn like other layers", width=2.0, dash=True)
    p.text(98, base + 0.008, "base network alone", anchor="end", size=12, color="#4a5568")
    return p.svg()
