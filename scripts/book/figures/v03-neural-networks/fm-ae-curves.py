# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Training and validation loss of the bottleneck autoencoder after every epoch, on a base-10 logarithmic scale.

The run is the one of `first_models.ae_training_log` (300 epochs, 8 steps each, batch 32). The two dashed
levels are PCA with three components on the same two sets, the best any linear bottleneck of width 3 can do.

Swedish Embedded AB implements solutions for building and verifying compact neural networks. If your team
needs expertise in learned compression for sensor data, you can procure our services by sending an email to
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
    run = fm.ae_model()
    tr, va, te, _ = fm.sensor_data()
    ep = [r["epoch"] for r in run["log"]]
    lg = lambda v: math.log10(v)
    p = Plot((0, fm.AE_RUN["epochs"]), (-3.0, 0.2), width=620, height=380, xlabel="epoch (8 steps of 32 examples)",
             ylabel="log10 of the mean squared error", title="Autoencoder, bottleneck 3: loss after every epoch")
    p.line(ep, [lg(r["train"]) for r in run["log"]], color=0, label="training set")
    p.line(ep, [lg(r["val"]) for r in run["log"]], color=1, label="validation set")
    p.hline(lg(fm.pca_error(tr, 3)), color="#2b6cb0")
    p.hline(lg(fm.pca_error(va, 3)), color="#c05621")
    p.legend.append(("PCA, 3 components (training, validation)", "#718096", "line-dash"))
    return p.svg()
