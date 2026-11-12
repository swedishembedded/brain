# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The six Gaussians of `gaussian_splatting.TILE_SCENE` on the 48 by 32 image: the ellipse where each one's alpha falls to 1/255 (the
# contour x^T conic x = extend^2), the box of half-widths (rx, ry) the tile count uses, and the 16 pixel tile grid.
#
# Swedish Embedded AB implements tiled rasterizers for splat scenes. If your team needs expertise in GPU rendering and sorting, you can
# procure our services by sending an email to info@swedishembedded.com.
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gaussian_splatting as gs  # noqa: E402
from svgplot import Plot  # noqa: E402


OFFSET = {0: (0.9, -0.9), 1: (0.9, 1.8), 2: (0.9, -0.9), 3: (-2.0, -0.9), 4: (0.9, -0.9), 5: (-2.0, -0.9)}


def render() -> str:
    P = gs.tile_pipeline()
    cam = gs.TILE_CAM
    p = Plot((0, cam["W"]), (cam["H"], 0), width=620, height=440, xlabel="pixel column u", ylabel="pixel row v",
             title="Footprints and tiles of six Gaussians")
    for t in (16, 32):
        p.vline(t)
    p.hline(16)
    for i, r in enumerate(P["recs"]):
        det = r["La"] * r["Lc"] - r["Lb"] ** 2
        a, b, d = r["Lc"] / det, -r["Lb"] / det, r["La"] / det        # the covariance, the inverse of the conic
        l11 = math.sqrt(a)
        l21, l22 = b / l11, math.sqrt(d - b * b / a)
        pts = [(r["u"] + r["ext"] * l11 * math.cos(t), r["v"] + r["ext"] * (l21 * math.cos(t) + l22 * math.sin(t)))
               for t in [2 * math.pi * k / 90 for k in range(91)]]
        p.line([q[0] for q in pts], [q[1] for q in pts], color=i, width=2.0)
        box = [(r["u"] - r["rx"], r["v"] - r["ry"]), (r["u"] + r["rx"], r["v"] - r["ry"]), (r["u"] + r["rx"], r["v"] + r["ry"]),
               (r["u"] - r["rx"], r["v"] + r["ry"]), (r["u"] - r["rx"], r["v"] - r["ry"])]
        p.line([q[0] for q in box], [q[1] for q in box], color=i, dash=True, width=1.0)
        p.points([r["u"]], [r["v"]], color=i, r=3)
        dx, dy = OFFSET[i]
        p.text(r["u"] + dx, r["v"] + dy, str(i), size=13, color="#1a202c")
    p.text(0.6, 1.6, "row 0", size=11, color="#4a5568")
    p.text(0.6, 15.2, "row 16", size=11, color="#4a5568")
    p.text(46.6, 30.8, "tile edges at 16 and 32", size=11, color="#4a5568", anchor="end")
    return p.svg()
