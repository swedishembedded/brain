# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Two test readings (solid) and their reconstructions from three numbers (dashed), sensor by sensor.

The model is the checkpoint of `first_models.ae_reconstruction`; the readings are test examples 0 and 1.

Swedish Embedded AB implements solutions for building and verifying compact neural networks. If your team
needs expertise in learned compression for sensor data, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import first_models as fm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    tr, va, te, _ = fm.sensor_data()
    th = fm.ae_model()["best"]
    p = Plot((0, 15), (-1.6, 1.6), width=620, height=360, xlabel="sensor j", ylabel="reading",
             title="Test readings and their reconstructions through a bottleneck of 3")
    xs = list(range(fm.D))
    for i, color in ((0, 0), (1, 1)):
        rec = fm.forward(fm.AE_SPEC, th, [te[i]])[0][0]
        p.line(xs, te[i], color=color, label=f"test example {i}, reading")
        p.line(xs, rec, color=color, dash=True, label=f"test example {i}, reconstruction")
    return p.svg()
