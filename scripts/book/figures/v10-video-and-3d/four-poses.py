# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The four decompositions of an essential matrix drawn from above, camera 0 at the bottom looking up: on which side camera 1 stands,
# which way it faces, and where the triangulated points land. Only one of them puts the points in front of both cameras. The side,
# the facing and the position of the points are the signs of the decomposition computed by `structure_from_motion.four_pose_figure_data`
# on the chapter's RANSAC inliers; the picture is a schematic and not to scale.
#
# Swedish Embedded AB implements solutions for structure from motion. If your team needs expertise in relative camera pose
# estimation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import structure_from_motion as sfm  # noqa: E402
from svgdiagram import Diagram, text_width  # noqa: E402

PANEL_W, PANEL_H, GAP, TOP = 300, 218, 16, 14
TITLES = ["R = U W V^T, t = +u3", "R = U W V^T, t = -u3", "R = U W^T V^T, t = +u3", "R = U W^T V^T, t = -u3"]


def render() -> str:
    data = sfm.four_pose_figure_data()
    caption_w = max(text_width(f"{c['front']} of {c['total']} points in front of both") for c in data)
    assert caption_w + 8 < PANEL_W - 28
    d = Diagram(2 * PANEL_W + 3 * GAP, TOP + 2 * PANEL_H + GAP + GAP)
    for k, c in enumerate(data):
        x0 = GAP + (k % 2) * (PANEL_W + GAP)
        y0 = TOP + (k // 2) * (PANEL_H + GAP)
        d.panel(x0, y0, PANEL_W, PANEL_H, TITLES[k])
        cx, cy = x0 + PANEL_W / 2, y0 + 116
        c0 = (cx - 40 if c["C"][0] > 0 else cx + 40, cy)       # camera 0 stands on the opposite side from camera 1
        c1 = (cx + 40 if c["C"][0] > 0 else cx - 40, cy)
        kind = "accent" if c["chosen"] else "plain"
        pt_up = c["mean"][2] > 0
        pt = d.oval(cx, cy - 58 if pt_up else cy + 52, 16, 16, [], "result")
        # rays from both centres to the point, then the cameras and their optical axes on top
        for cam in (c0, c1):
            d.line(cam, pt["b"] if pt_up else pt["t"])
        b0 = d.box(*c0, 34, 24, ["0"], "model")
        b1 = d.box(*c1, 34, 24, ["1"], kind)
        d.arrow(b0["t"], (c0[0], cy - 44))
        up1 = c["axis"][2] > 0
        d.arrow(b1["t"] if up1 else b1["b"], (c1[0], cy - 44 if up1 else cy + 44))
        d.text(cx, y0 + PANEL_H - 22, [f"{c['front']} of {c['total']} points in front of both"], size=13,
               color="#c53030" if not c["chosen"] else "#1a202c")
    return d.svg()
