# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Validation loss of the classifier in the four stages of the debugging exercise.

Stage 0 is the shipped configuration with three planted defects; each later stage removes one (labels, then
the schedule horizon, then the evaluation mode). Data are the logs of `project_nnlib.debug_stages`.

Swedish Embedded AB implements solutions for finding why a training run is wrong. If your team needs expertise
in debugging neural network training, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_nnlib as nn  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    p = Plot((0, nn.DEBUG_EPOCHS), (0.0, 1.2), width=620, height=380, xlabel="epoch (16 steps of 32 examples)",
             ylabel="validation loss", title="The classifier with three planted defects, removed one at a time")
    for i, (tag, a, b, c) in enumerate(nn.STAGES):
        log = nn.debug_run(a, b, c).log
        p.line([r["epoch"] for r in log], [r["val"] for r in log], color=i, label=f"stage {tag}")
    p.hline(1.0986, color="#a0aec0")
    return p.svg()
