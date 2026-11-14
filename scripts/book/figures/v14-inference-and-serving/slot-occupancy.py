# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Which request holds which slot of a four-row decode batch, iteration by iteration, under static and under continuous
# batching, in the chapter "Batching and Scheduling Requests". The schedules are the worked module's `static_slots` and
# `continuous_slots`, so the picture cannot disagree with the table beside it.
#
# Swedish Embedded AB implements solutions for serving language models to many users at once on the hardware they already
# have. If your team needs expertise in continuous batching then you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import batching as bt  # noqa: E402
from svgdiagram import FRAME, MUTED, Diagram  # noqa: E402
from svgplot import FONT, fmt  # noqa: E402

W, H = 1000, 450
X0 = 120                  # x of the start of iteration 1
PX = 10                   # px per iteration
ROW = 28                  # px per slot row
TOPS = {"static": 62, "continuous": 262}


def render() -> str:
    d = Diagram(W, H)
    st = bt.static_slots(bt.LENGTHS, bt.SLOTS)
    co = bt.continuous_slots(bt.LENGTHS, bt.SLOTS)
    end = max(last for _, _, last in st)
    for name, sched in (("static", st), ("continuous", co)):
        top = TOPS[name]
        done = max(last for _, _, last in sched)
        d.panel(20, top - 42, W - 40, 4 * ROW + 70, f"{name} batching: {done} iterations for {sum(bt.LENGTHS)} tokens on 4 slots")
        for s in range(bt.SLOTS):
            d.text(X0 - 14, top + s * ROW + ROW / 2, [f"slot {s}"], size=12, color=MUTED, anchor="end")
        if name == "static":
            # A batch's idle slots: from a member's last token to the end of the batch.
            t = 1
            for b in range(0, len(bt.LENGTHS), bt.SLOTS):
                batch = bt.LENGTHS[b:b + bt.SLOTS]
                stop = t + max(batch) - 1
                for j, n in enumerate(batch):
                    if t + n - 1 < stop:
                        x = X0 + (t + n - 1) * PX
                        d.items.append(f'<rect x="{fmt(x)}" y="{fmt(top + j * ROW + 3)}" width="{fmt((stop - (t + n - 1)) * PX)}" '
                                       f'height="{ROW - 6}" fill="none" stroke="{FRAME}" stroke-width="1.2" stroke-dasharray="4 3"/>')
                t = stop + 1
        for i, (s, a, b) in enumerate(sched):
            x = X0 + (a - 1) * PX
            w = (b - a + 1) * PX
            d.items.append(f'<rect x="{fmt(x + 1)}" y="{fmt(top + s * ROW + 3)}" width="{fmt(w - 2)}" height="{ROW - 6}" rx="4" '
                           f'fill="#ebf4ff" stroke="{MUTED}" stroke-width="1.1"/>')
            d.text(x + w / 2, top + s * ROW + ROW / 2, [str(i)], size=12)
        axis = top + 4 * ROW + 6
        d.items.append(f'<line x1="{X0}" y1="{fmt(axis)}" x2="{fmt(X0 + end * PX)}" y2="{fmt(axis)}" stroke="{MUTED}" stroke-width="1"/>')
        for k in (1, 20, 40, 60, 80):
            x = X0 + (k - 0.5) * PX
            d.items.append(f'<line x1="{fmt(x)}" y1="{fmt(axis)}" x2="{fmt(x)}" y2="{fmt(axis + 4)}" stroke="{MUTED}" stroke-width="1"/>')
            d.text(x, axis + 13, [str(k)], size=11, color=MUTED)
    d.text(X0 + end * PX / 2, H - 12, ["iteration (each box is one request, labelled by its number; dashed: a slot held idle)"],
           size=12, color=MUTED)
    return d.svg()
