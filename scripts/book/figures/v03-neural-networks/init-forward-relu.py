# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Root mean square of the pre-activation z_l, layer by layer, of the ten-layer ReLU stack from four starts (computed by
worked module `initialization`, function `figure_forward`). The all-zero start is 0 everywhere and is not on a log scale.
The dashed line is rms 1.

Swedish Embedded AB implements solutions for making deep networks train from their first step. If your team needs
expertise in weight initialization and activation statistics, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import initialization as ini  # noqa: E402
from svgplot import Plot  # noqa: E402

LABELS = {"small": "small (0.01)", "large": "large (1)", "xavier": "Xavier", "he": "He"}


def render() -> str:
    series = ini.figure_forward("relu")
    logs = {k: [math.log10(v) for v in vs] for k, vs in series.items()}
    lo = math.floor(min(min(v) for v in logs.values())) - 1
    hi = math.ceil(max(max(v) for v in logs.values())) + 1
    p = Plot((1, ini.DEPTH), (lo, hi), width=620, height=380, xlabel="layer l", ylabel="log10 of the rms of z_l",
             title="ReLU units: size of the pre-activation z_l by layer")
    p.hline(0.0)
    for i, (scheme, vals) in enumerate(logs.items()):
        p.line(list(range(1, ini.DEPTH + 1)), vals, color=i, width=2.0, label=LABELS[scheme])
    return p.svg()
