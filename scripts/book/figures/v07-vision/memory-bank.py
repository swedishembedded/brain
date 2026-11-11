# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The memory bank of the video path at frame 12 of a clip prompted on frame 0: the frames whose memory slabs the current frame reads
# (the prompted frame and the six before it, with their temporal positions) and the frames that contribute an object pointer
# (`segment_anything.bank_figure_data`, which applies the rule of `segment_anything.frame_memory`).
#
# Swedish Embedded AB implements solutions for tracking an object through video with bounded memory. If your team needs expertise in
# video object segmentation, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import segment_anything as sa  # noqa: E402
from svgdiagram import FRAME, INK, MUTED, Diagram  # noqa: E402

W, H = 940, 400
BOX, STEP, X0, Y = 66, 68, 40, 120


def render() -> str:
    data = sa.bank_figure_data(12)
    f, slabs, ptrs = data["frame"], dict((src, tp) for tp, src in data["slabs"]), set(data["ptr_frames"])
    d = Diagram(W, H)
    d.text(W / 2, 30, [f"Frame {f} reads memory slabs of {len(slabs)} frames and object pointers of {len(ptrs)} frames"], size=14, weight="bold")
    cx = lambda t: X0 + t * STEP + BOX / 2
    for t in range(f + 1):
        if t == f:
            kind, lab = "result", [f"frame {t}", "current"]
        elif t == 0:
            kind, lab = "accent", [f"frame {t}", "prompted", "slot 0"]
        elif t in slabs:
            kind, lab = "model", [f"frame {t}", f"slot {slabs[t]}"]
        else:
            kind, lab = "plain", [f"frame {t}", "pointer"]
        d.box(cx(t), Y, BOX, 78, lab, kind)
    bank = d.box(W / 2 - 150, 290, 300, 84, ["memory tokens", f"{len(slabs)} slabs x 4,096 + {len(ptrs)} pointers x 4", f"= {data['keys']:,} keys"], "accent")
    cur = d.box(W / 2 + 200, 290, 280, 84, ["memory attention", "4,096 queries (frame 12)"], "model")
    for t in slabs:
        d.line((cx(t), Y + 39), (W / 2 - 150, 248), dash="3 3")
    d.arrow(bank["r"], cur["l"])
    d.path([(cx(f), Y + 39), (cx(f), 200), (cur["t"][0], 200), (cur["t"][0], cur["t"][1])], radius=8)
    d.text(790, 232, ["the current frame's", "features are the queries"], size=12, color=MUTED)
    d.text(W / 2, H - 14, ["Every frame before 12 also lends an object pointer (12 pointers). Slot 6 is the previous frame, slot 1 "
                           "six frames back; the prompted frame is always read, as slot 0."], size=12, color=MUTED)
    return d.svg()
