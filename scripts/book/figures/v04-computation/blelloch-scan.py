# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The Blelloch scan of the eight numbers of Volume I, Chapter 16, as the kernel of the chapter runs it: an up-sweep that
# builds partial sums in place (7 additions), the root cleared to 0, and a down-sweep that passes prefixes down (7
# additions). Every row is the shared array at a barrier, taken from the worked module's simulator run; an arrow is
# drawn for each value that is read to produce a changed value. Solid: added into the target. Dashed: copied.
#
# Swedish Embedded AB implements solutions for parallel prefix sums on GPUs. If your team needs expertise in scan and
# sort kernels for machine learning, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import reductions as rd  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

N = len(rd.X8)
PITCH, BOX_W, BOX_H, ROW = 66, 46, 28, 68
X0 = 212
Y0 = 70
W = X0 + N * PITCH + 120
H = Y0 + 7 * ROW + 50


def _centre(col):
    return X0 + (col + 0.5) * PITCH


def render() -> str:
    d = Diagram(W, H)
    snaps = rd.scan_snapshots(rd.X8, "blelloch")
    ups, downs = rd.STRIDES_UP(N), rd.STRIDES_DOWN(N)
    d.panel(14, 12, W - 28, H - 24, "Blelloch scan of eight numbers: 7 additions up, 7 down")
    boxes = []
    for r, (label, arr) in enumerate(snaps):
        y = Y0 + r * ROW
        prev = snaps[r - 1][1] if r else None
        row = []
        for c, v in enumerate(arr):
            changed = prev is not None and prev[c] != v
            kind = "result" if r == len(snaps) - 1 else ("model" if changed else "plain")
            row.append(d.box(_centre(c), y, BOX_W, BOX_H, [str(int(v))], kind))
        boxes.append(row)
        d.text(34, y, [label], size=12, color="#4a5568", anchor="start")
    for r in range(1, len(snaps)):
        if 1 <= r <= len(ups):                       # up-sweep: a[i] = a[i] + a[i - d]
            dd = ups[r - 1]
            for l in range(N // (2 * dd)):
                i = 2 * dd * (l + 1) - 1
                d.arrow(boxes[r - 1][i]["b"], boxes[r][i]["t"])
                d.arrow(boxes[r - 1][i - dd]["b"], boxes[r][i]["t"])
            d.text(X0 + N * PITCH + 14, Y0 + r * ROW, [f"{N // (2 * dd)} addition" + ("" if N == 2 * dd else "s")], size=12, color="#4a5568", anchor="start")
        elif r > len(ups) + 1:                       # down-sweep: t = a[i - d]; a[i - d] = a[i]; a[i] = a[i] + t
            dd = downs[r - len(ups) - 2]
            for l in range(N // (2 * dd)):
                i = 2 * dd * (l + 1) - 1
                d.arrow(boxes[r - 1][i]["b"], boxes[r][i]["t"])
                d.arrow(boxes[r - 1][i - dd]["b"], boxes[r][i]["t"])
                d.arrow(boxes[r - 1][i]["b"], boxes[r][i - dd]["t"], dash="4 3")
            d.text(X0 + N * PITCH + 14, Y0 + r * ROW, [f"{N // (2 * dd)} addition" + ("" if N == 2 * dd else "s")], size=12, color="#4a5568", anchor="start")
    return d.svg()
