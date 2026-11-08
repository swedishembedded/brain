# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Relative output error (divided by its value for the min and max range) of the quantized network on spiky inputs against the tail clipped from each end of the calibrated
# ranges: the typical input (median) improves until about one percent is clipped and then worsens, while the rare large
# inputs (99th percentile) only get worse. The data is the worked module's `clipping_curve`, the table `npu.clipping`.
#
# Swedish Embedded AB implements solutions for choosing calibration ranges for integer accelerators. If your team needs
# expertise in activation range selection for post-training quantization, you can procure our services by sending an
# email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import npu  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    curve = npu.clipping_curve()
    xs = [100 * c[0] for c in curve]
    med0, p990 = curve[0][2], curve[0][3]
    p = Plot((0.0, 5.0), (0.0, 2.4), width=600, height=360, xlabel="tail clipped from each end of every range, percent",
             ylabel="relative error, 1 = min and max range", title="Clipping trades the rare large input for the typical one")
    p.hline(1.0)
    p.line(xs, [c[2] / med0 for c in curve], color=0, label="median input", width=2.0)
    p.points(xs, [c[2] / med0 for c in curve], color=0, r=3.5)
    p.line(xs, [c[3] / p990 for c in curve], color=1, label="99th percentile input", width=2.0)
    p.points(xs, [c[3] / p990 for c in curve], color=1, r=3.5)
    return p.svg()
