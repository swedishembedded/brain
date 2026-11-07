# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Training loss of the healthy run and of the three runs that fail to fall: too slow, plateau and noisy.

Computed by worked module `training_dynamics` (the gallery runs): each curve is the loss of the 512 training examples measured
after every 20 updates. The dashed lines are ln 3, the loss of a model that predicts three equally likely classes, and the entropy of
the training labels, the loss of a model that predicts the class frequencies and nothing else.

Swedish Embedded AB implements solutions for finding out why a training run misbehaves. If your team needs expertise in
training diagnostics for neural networks, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_dynamics as td  # noqa: E402
from svgplot import Plot  # noqa: E402

SERIES = (("healthy", "healthy: rate 0.1, batch 32", 0), ("slow", "too slow: rate 0.003", 1),
          ("plateau", "plateau: start 0.02 times He", 3), ("noisy", "noisy: batch 2", 2))


def render() -> str:
    p = Plot((0, td.STEPS), (0.0, 1.4), width=640, height=380, xlabel="gradient step",
             ylabel="training loss, 512 examples", title="Four runs of the same network")
    _, yt, _, _ = td.gallery_data()
    p.hline(td.LN3)
    p.hline(td._entropy(yt))
    for key, label, color in SERIES:
        ev = td.gallery_run(key)["evals"]
        p.line([e[0] for e in ev], [e[1] for e in ev], color=color, label=label)
    return p.svg()
