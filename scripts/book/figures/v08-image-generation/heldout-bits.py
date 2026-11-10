# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Held-out bits per image of three models of the chapter's world as the training set grows, against the entropy of the world, the
# floor no model can pass. The counting table is absent: its held-out bits are infinite until every image has been seen. The data
# are `generative_modeling.heldout_curve`, the arithmetic of the chapter's table.
#
# Swedish Embedded AB implements solutions for generative models that are judged on data they have not seen. If your team needs
# expertise in density estimation for images, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import generative_modeling as gm  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    sizes, series, entropy = gm.heldout_curve()
    p = Plot((10, 1000), (3.0, 4.9), width=560, height=360, xlabel="training images N", ylabel="held-out bits per image",
             title="Structure buys data", legend_loc="ur", xlog=True)
    p.line([10, 1000], [entropy, entropy], color=5, label=f"entropy of the world, {entropy:.3f}", dash=True, width=1.4)
    for k, (name, ys) in enumerate(series.items()):
        p.line(sizes, ys, color=k, label=name, width=2.0)
        p.points(sizes, ys, color=k, r=3)
    return p.svg()
