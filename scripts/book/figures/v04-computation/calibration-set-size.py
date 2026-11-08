# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Output SQNR of the quantized two-layer network against the number of calibration inputs, for a representative
# calibration stream and for a stream a quarter of the size of the deployed inputs. The data is the worked module's
# `calibration_curve`, the table `npu.calibration_size`.
#
# Swedish Embedded AB implements solutions for calibrating neural networks for integer accelerators. If your team needs
# expertise in post-training quantization, you can procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import npu  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curve = npu.calibration_curve()
    xs = [math.log2(n) for n, *_ in curve]
    p = Plot((0.0, 7.0), (0.0, 30.0), width=600, height=360, xlabel="log2 of the number of calibration inputs",
             ylabel="SQNR of the output in dB", title="A calibration set must show the network what it will see")
    p.line(xs, [c[1] for c in curve], color=0, label="inputs of the deployed size", width=2.0)
    p.points(xs, [c[1] for c in curve], color=0, r=3.5)
    p.line(xs, [c[3] for c in curve], color=1, label="inputs a quarter of that size", width=2.0)
    p.points(xs, [c[3] for c in curve], color=1, r=3.5)
    return p.svg()
