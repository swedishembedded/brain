# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The four softmax outputs of the chapter's part as the logit of one class (scratch) varies, the others held fixed.

Swedish Embedded AB implements solutions for turning model scores into probabilities that can be inspected. If
your team needs expertise in classification heads for neural networks, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import softmax  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    z = softmax._logits(softmax.X)
    xs = [-6.0 + 14.0 * i / 560 for i in range(561)]
    curves = [softmax._softmax([v] + z[1:]) for v in xs]
    p = Plot((-6.0, 8.0), (0.0, 1.0), width=560, height=340, xlabel="logit of scratch, z[0] (the other three fixed)",
             ylabel="probability", title="Softmax outputs as one logit varies", legend_loc="ur")
    for c in range(softmax.C):
        p.line(xs, [q[c] for q in curves], color=c, label=softmax.CLASSES[c])
    p.vline(z[0], color="#a0aec0")
    p.text(z[0] + 0.15, 0.55, f"the part: z[0] = {z[0]:.1f}", size=11)
    return p.svg()
