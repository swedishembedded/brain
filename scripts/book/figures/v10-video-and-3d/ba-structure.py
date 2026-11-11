# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The sparsity of a bundle adjustment: the Jacobian of the residuals of three cameras and four points (10 observations, 2 rows each),
# the normal-equations matrix J^T J, in which the camera blocks and the point blocks are block-diagonal and only the observations couple
# them, and the reduced camera system left after the points are eliminated, which is dense in the camera blocks that share a point.
# The pattern is computed from the list of observations below.
#
# Swedish Embedded AB implements solutions for bundle adjustment and large least-squares problems. If your team needs expertise in
# sparse nonlinear least squares, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402

OBS = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2), (2, 2), (0, 3), (2, 3), (1, 3)]   # (camera, point), 3 cameras, 4 points
NC, NP = 3, 4
U = 7                       # pixels per matrix column
CAM_W, PT_W = 6, 3          # columns per camera block and per point block
FILL, EDGE = "#2b6cb0", "#a0aec0"


def cells(d, x, y, blocks):
    """Filled rectangles for (row0, col0, rows, cols) blocks of the matrix whose top-left corner is (x, y)."""
    for r, c, h, w in blocks:
        d.items.append(f'<rect x="{x + c * U}" y="{y + r * U}" width="{w * U}" height="{h * U}" fill="{FILL}" stroke="#ffffff" '
                       f'stroke-width="0.6"/>')


def outline(d, x, y, rows, cols):
    d.items.append(f'<rect x="{x}" y="{y}" width="{cols * U}" height="{rows * U}" fill="none" stroke="{EDGE}" stroke-width="1"/>')


def render() -> str:
    ncols = NC * CAM_W + NP * PT_W            # 30
    nrows = 2 * len(OBS)                      # 20
    pw = ncols * U + 28                       # panel width
    gap = 18
    H0 = 70 + max(nrows, ncols) * U + 34
    d = Diagram(3 * pw + 4 * gap - 40, H0 + 8)
    # (a) Jacobian
    xa, ya = gap, 10
    d.panel(xa, ya, pw, H0 - 10, "J: 20 residuals by 30 unknowns")
    ox, oy = xa + 14, ya + 62
    blocks = []
    for k, (cam, pt) in enumerate(OBS):
        blocks.append((2 * k, cam * CAM_W, 2, CAM_W))
        blocks.append((2 * k, NC * CAM_W + pt * PT_W, 2, PT_W))
    cells(d, ox, oy, blocks)
    outline(d, ox, oy, nrows, ncols)
    for cam in range(NC):
        d.text(ox + (cam * CAM_W + CAM_W / 2) * U, oy - 10, [f"c{cam}"], size=11, color="#4a5568")
    for pt in range(NP):
        d.text(ox + (NC * CAM_W + pt * PT_W + PT_W / 2) * U, oy - 10, [f"p{pt}"], size=11, color="#4a5568")
    # (b) J^T J
    xb = xa + pw + gap
    d.panel(xb, ya, pw, H0 - 10, "J^T J: 30 by 30")
    ox, oy = xb + 14, ya + 62
    blocks = [(cam * CAM_W, cam * CAM_W, CAM_W, CAM_W) for cam in range(NC)]
    blocks += [(NC * CAM_W + pt * PT_W, NC * CAM_W + pt * PT_W, PT_W, PT_W) for pt in range(NP)]
    for cam, pt in OBS:
        blocks.append((cam * CAM_W, NC * CAM_W + pt * PT_W, CAM_W, PT_W))
        blocks.append((NC * CAM_W + pt * PT_W, cam * CAM_W, PT_W, CAM_W))
    cells(d, ox, oy, blocks)
    outline(d, ox, oy, ncols, ncols)
    for cam in range(NC):
        d.text(ox + (cam * CAM_W + CAM_W / 2) * U, oy - 10, [f"c{cam}"], size=11, color="#4a5568")
    for pt in range(NP):
        d.text(ox + (NC * CAM_W + pt * PT_W + PT_W / 2) * U, oy - 10, [f"p{pt}"], size=11, color="#4a5568")
    # (c) the reduced camera system
    xc = xb + pw + gap
    w3 = NC * CAM_W * U + 28
    d.panel(xc, ya, w3 + 22, H0 - 10, "Schur: 18 by 18")
    ox, oy = xc + 25, ya + 62
    blocks = []
    for a in range(NC):
        for b in range(NC):
            shared = any(OBS_has(a, p) and OBS_has(b, p) for p in range(NP))
            if shared:
                blocks.append((a * CAM_W, b * CAM_W, CAM_W, CAM_W))
    cells(d, ox, oy, blocks)
    outline(d, ox, oy, NC * CAM_W, NC * CAM_W)
    for cam in range(NC):
        d.text(ox + (cam * CAM_W + CAM_W / 2) * U, oy - 10, [f"c{cam}"], size=11, color="#4a5568")
    return d.svg()


def OBS_has(cam, pt):
    return (cam, pt) in OBS
