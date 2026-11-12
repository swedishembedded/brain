# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# A view from above of the table of the project: the reconstructed table and its clamp box, the detected paths of the three balls
# over the 12 observed frames, their true paths for the next six, and the two predictions (clamped and not) of the slider that is
# stopped by the table's edge.
#
# Swedish Embedded AB implements solutions for predicting where a tracked object will be. If your team needs expertise in learned
# dynamics for moving objects in reconstructed scenes, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_dynamic_scene as pds  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    d = pds._floor_plan()
    t, b = d["table"], d["box"]
    p = Plot((-1.4, 1.9), (3.8, 6.6), width=560, height=420, xlabel="x (metres, right)", ylabel="z (metres, away from the camera)",
             title="The table from above")
    p.line([t[0], t[1], t[1], t[0], t[0]], [t[2], t[2], t[3], t[3], t[2]], color="#4a5568", width=1.4, label="table")
    p.line([b[0], b[1], b[1], b[0], b[0]], [b[2], b[2], b[3], b[3], b[2]], color="#4a5568", dash=True, width=1.2, label="clamp box")
    for j, name in enumerate("ABC"):
        det = d["det"][j]
        p.points([q[0] for q in det], [q[1] for q in det], color=j, r=3.0, label=f"{name}, detected")
        tr = d["truth"][j][pds.OBSERVED - 1:]
        p.line([q[0] for q in tr], [q[1] for q in tr], color=j, width=1.6)
    p.text(-1.08, 4.1, "solid lines: true paths from frame 11 to frame 17", size=11)
    free = [d["det"][2][-1]] + list(d["free"][2])
    p.line([q[0] for q in free], [q[1] for q in free], color=4, dash=True, width=1.6, label="C, unclamped")
    return p.svg()
