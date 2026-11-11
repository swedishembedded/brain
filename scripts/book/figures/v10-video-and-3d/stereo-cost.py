# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The photometric cost (1 - weighted normalized cross-correlation, mean over two source views) of the plane hypotheses through one
# reference pixel, as the range along its ray varies: with a fronto-parallel normal and with the true normal of the tilted plane. The
# data are `structure_from_motion.sweep_curve` on the chapter's three small images.
#
# Swedish Embedded AB implements solutions for dense reconstruction from photographs. If your team needs expertise in multi-view
# stereo, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import structure_from_motion as sfm  # noqa: E402
from svgplot import Plot  # noqa: E402

PIXEL = (36, 26)


def render() -> str:
    front = sfm.sweep_curve(*PIXEL, rmax=5.5, samples=160)
    true = sfm.sweep_curve(*PIXEL, sfm.MVS_NORMAL, rmax=5.5, samples=160)
    rt = sfm.true_range(PIXEL[0] + 0.5, PIXEL[1] + 0.5)
    p = Plot((2.5, 5.5), (0.0, 1.0), width=560, height=340, xlabel="hypothesised range along the ray (m)",
             ylabel="cost, 1 - NCC", title="Plane sweep through pixel (36, 26)", legend_loc="ur")
    p.line([r for r, c in front], [c for r, c in front], color=0, label="fronto-parallel normal", width=1.9)
    p.line([r for r, c in true], [c for r, c in true], color=2, label="true normal", width=1.9)
    p.vline(rt)
    p.text(4.3, 0.5, "dashed line: true range %.2f" % rt, anchor="start")
    return p.svg()
