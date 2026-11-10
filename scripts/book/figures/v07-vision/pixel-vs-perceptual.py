# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Pixel mean squared error against the feature distance of the toy perceptual metric (`clip.perceptual_data`) for the distortions of the chapter,
# on logarithmic axes: each series is one kind of distortion at increasing strength.
#
# Swedish Embedded AB implements solutions for image-quality evaluation that tracks what a viewer sees. If your team needs expertise in
# perceptual metrics, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import clip  # noqa: E402
from svgplot import Plot  # noqa: E402

NAMES = {"shift": "shifted by 1", "blur": "blurred 1, 2, 3 times", "noise": "noise, sd 0.05 to 0.3", "contrast": "contrast 0.75, 0.5, 0.25"}


def render() -> str:
    rows = clip.perceptual_data()
    p = Plot((0.002, 0.3), (0.01, 3.0), width=600, height=360, xlabel="pixel mean squared error", ylabel="feature distance (taps summed)",
             legend_loc="auto", xlog=True, ylog=True)
    for k, series in enumerate(("shift", "blur", "noise", "contrast")):
        pts = [(r[2], r[3] + r[4]) for r in rows if r[0] == series]
        if len(pts) > 1:
            p.line([a for a, _ in pts], [b for _, b in pts], color=k, width=1.5)
        p.points([a for a, _ in pts], [b for _, b in pts], color=k, label=NAMES[series], r=4.5)
    return p.svg()
