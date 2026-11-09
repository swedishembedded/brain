# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The sRGB transfer function dec (encoded value to linear light) against the straight line a linear encoding would be, with the
# code 128 marked. The curve is `images.curve_data`, the function the chapter's tables are computed from.
#
# Swedish Embedded AB implements solutions for colour-correct image pipelines. If your team needs expertise in image encoding and
# colour management for vision models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import images as im  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    xs, ys = im.curve_data(201)
    p = Plot((0, 1), (0, 1), width=560, height=400, xlabel="encoded value E = code / 255",
             ylabel="linear light dec(E)", title="The sRGB code is not proportional to light", legend_loc="ul")
    p.line([0, 1], [0, 1], color=3, label="proportional to light", dash=True, width=1.4)
    p.line(xs, ys, color=0, label="sRGB: dec(E)", width=2.0)
    e = 128 / 255
    p.vline(e)
    p.hline(im.dec(e))
    p.points([e], [im.dec(e)], color=1, r=4)
    p.text(e + 0.02, im.dec(e) - 0.06, f"code 128: light {im.dec(e):.3f}", size=11)
    return p.svg()
