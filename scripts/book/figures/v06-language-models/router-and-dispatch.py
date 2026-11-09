# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The data flow of one mixture-of-experts layer: the router turns each token into k (expert, gate weight) slots, a counting sort groups
# the slots by expert, every expert runs once on the rows routed to it, and the weighted results are added back to their tokens next
# to the shared expert's output. The row counts in the expert boxes are those of the dispatch example of the chapter
# (`mixture_of_experts.dispatch_trace`).
#
# Swedish Embedded AB implements solutions for serving sparse language models. If your team needs expertise in mixture-of-experts
# dispatch kernels, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mixture_of_experts as moe  # noqa: E402
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 610


def render() -> str:
    s = moe.dispatch_setup()
    slot_expert = [e for chosen in s["ids"] for e in chosen]
    counts, _, _ = moe.histogram_scan_emit(slot_expert, moe.DISP_E)
    d = Diagram(W, H)

    x = d.oval(130, 70, 170, 56, ["x", "n tokens, width D"], "accent")
    router = d.box(400, 70, 240, 58, ["router", "z = x W_r, softmax,", "top-k, renormalize"])
    slots = d.oval(760, 70, 210, 58, ["n k slots:", "expert id, gate weight"], "result")
    sort = d.box(760, 200, 230, 58, ["counting sort of the slots", "histogram, scan, emit"])
    gather = d.box(400, 200, 240, 58, ["gather the rows of x", "expert by expert"])
    d.arrow(x["r"], router["l"])
    d.arrow(router["r"], slots["l"])
    d.arrow(slots["b"], sort["t"])
    d.arrow(sort["l"], gather["r"], label="expert-major order")
    d.path([(x["b"][0], x["b"][1]), (x["b"][0], 200), gather["l"]])

    gx, gy, gw, gh = 30, 290, 740, 100
    grp = d.group(gx, gy, gw, gh)
    d.text(gx + gw / 2, gy + 14, ["every expert runs once, on its own rows (the example's counts)"], size=12, color=MUTED)
    bw = 160
    for e in range(moe.DISP_E):
        cx = gx + 20 + bw / 2 + e * (bw + 20)
        d.box(cx, gy + 62, bw, 48, [f"expert {e}", f"{counts[e]} rows"])
    d.arrow(gather["b"], grp["t"])

    shared = d.box(865, 340, 180, 76, ["shared expert", "reads all n rows,", "no router"])
    combine = d.box(400, 480, 340, 62, ["scatter back and add", "out[token] += g * expert output,", "plus the shared expert's output"])
    d.arrow(grp["b"], combine["t"])
    d.path([(slots["r"][0], slots["r"][1]), (965, slots["r"][1]), (965, 466), (combine["r"][0], 466)])
    d.path([(shared["b"][0], shared["b"][1]), (shared["b"][0], 494), (combine["r"][0], 494)])
    d.text(952, 250, ["gate weights g"], size=12, color=MUTED, anchor="end")
    y = d.oval(400, 570, 220, 44, ["y: n tokens, width D"], "accent")
    d.arrow(combine["b"], y["t"])
    return d.svg()
