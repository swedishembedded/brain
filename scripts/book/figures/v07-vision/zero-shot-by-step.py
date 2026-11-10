# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Zero-shot accuracy against the training step of the contrastive run (`clip.training_curve_data`): the 12 training classes of run 0, and the 4
# held-out classes of the three runs (initializations `clip.init.0`, `clip.init.1`, `clip.init.2`), with the chance level 1/16.
#
# Swedish Embedded AB implements solutions for measuring what a joint image and text embedding actually generalizes to. If your team needs
# expertise in zero-shot evaluation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import clip  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    d = clip.training_curve_data()
    steps = d["steps"]
    p = Plot((0, steps[-1]), (0, 1.0), width=600, height=340, xlabel="training step (12 pairs per step)",
             ylabel="zero-shot accuracy over 16 class prompts", legend_loc="auto")
    p.line(steps, d["seen"], color=0, label="training classes (run 0)")
    for r, held in enumerate(d["held_runs"]):
        p.line(steps, held, color=[1, 2, 4][r], label=f"held-out classes (run {r})")
    p.hline(1 / 16)
    p.text(steps[-1] * 0.99, 1 / 16 + 0.035, "chance 1/16", anchor="end", size=11, color="#4a5568")
    return p.svg()
