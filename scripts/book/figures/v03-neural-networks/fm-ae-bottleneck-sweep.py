# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Test error against the width of the bottleneck: PCA, the trained linear autoencoders and the trained nonlinear one.

PCA's error for k = 0 .. 7 is the sum of the discarded eigenvalues over 16 (`first_models.pca_spectrum`); the linear
autoencoders of widths 1 .. 4 and the nonlinear one of width 3 are the runs of `first_models.linear_vs_pca` and
`first_models.ae_reconstruction`.

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
    tr, va, te, _ = fm.sensor_data()
    lg = math.log10
    ks = list(range(0, 8))
    pca = [fm.mean_predictor_loss(te, tr) if k == 0 else fm.pca_error(te, k) for k in ks]
    p = Plot((0, 7), (-3.0, 0.0), width=620, height=360, xlabel="bottleneck width k",
             ylabel="log10 of the test error", title="Test error against the width of the bottleneck")
    p.line(ks, [lg(v) for v in pca], color=0, label="PCA with k components")
    lin_k = [1, 2, 3, 4]
    p.points(lin_k, [lg(fm.loss_only(fm.linear_spec(k), fm.linear_model(k)["best"], te, te)) for k in lin_k], color=1, label="linear autoencoder", r=4.5)
    p.points([3], [lg(fm.loss_only(fm.AE_SPEC, fm.ae_model()["best"], te, te))], color=2, label="autoencoder with GELU layers", r=4.5)
    p.hline(lg(fm.SENSOR_NOISE ** 2 * 13 / 16))
    return p.svg()
