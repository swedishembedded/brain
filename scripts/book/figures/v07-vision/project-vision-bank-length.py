# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Mask IoU with the truth of the trackers of objects A and C through the clip of the Volume VII project, with the full bank (the prompt
# frame and 6 recent frames) and with a bank of the prompt frame and 2 recent frames (`project_vision_pipeline.iou_series`). Object A
# is hidden behind B on frames 7 and 8, so its curves have a gap there. The short bank cannot reach back over the gap: A is lost when
# it comes out, and C, which looks the same as A, takes A's pixels as its own.
#
# Swedish Embedded AB implements solutions for tracking objects through occlusion with a bounded memory. If your team needs expertise
# in video object tracking, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import project_vision_pipeline as pv  # noqa: E402
from svgplot import Plot  # noqa: E402

# object, recent frames in the bank, label, colour, dashed
SERIES = (("A", 6, "A, bank of 6 recent frames", 0, False), ("A", 2, "A, bank of 2 recent frames", 3, True),
          ("C", 6, "C, bank of 6 recent frames", 1, False), ("C", 2, "C, bank of 2 recent frames", 2, True))


def render() -> str:
    p = Plot((0, 11), (0, 1.05), width=620, height=380, xlabel="frame", ylabel="mask IoU with the truth",
             title="A short bank loses A after it is hidden and lets C absorb it", legend_loc="below")
    for name, n, label, color, dash in SERIES:
        ys = pv.iou_series(name, n_recent=n)
        runs = ((0, 6), (9, 11)) if name == "A" else ((0, 11),)
        for i, (lo, hi) in enumerate(runs):
            xs = list(range(lo, hi + 1))
            p.line(xs, [ys[t] for t in xs], color=color, label=label if i == 0 else "", dash=dash)
            p.points(xs, [ys[t] for t in xs], color=color, r=3)
    p.text(7.5, 0.55, "A hidden", anchor="middle", size=12, color="#4a5568", plate=True)
    return p.svg()
