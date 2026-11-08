# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The programming hierarchy of a GPU kernel for n = 1000 elements: the dispatch grid of 16 workgroups, one
# workgroup of 64 invocations split into two subgroups of 32 and sharing a shared memory, and one subgroup
# with its 32 lanes, each lane an invocation with its own registers. The counts and the global indices come
# from the worked module's `grid_shape`, so the picture and the table of the chapter cannot disagree.
#
# Swedish Embedded AB implements solutions for mapping machine learning kernels onto the thread hierarchy of
# a GPU. If your team needs expertise in GPU kernel design for inference and training, you can procure our
# services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gpu  # noqa: E402
from svgdiagram import Diagram  # noqa: E402

W, MARGIN = 780, 20
PANEL_H, GAP, TOP = 118, 34, 14
PANEL2_H = 152
BOX_H = 50
SLOTS = (95, 235, 375, 515, 685)      # centres of five boxes across a panel
BOX_W = 118


def render() -> str:
    groups, _, _, per_wg = gpu.grid_shape()
    last = groups - 1
    wg = gpu.GRID_WG
    sg = gpu.SUBGROUP
    first_idx = last * wg                       # global index of the first invocation of the last workgroup
    sub1 = first_idx + sg                       # first invocation of its subgroup 1
    h = TOP + 2 * PANEL_H + PANEL2_H + 2 * GAP + 14
    d = Diagram(W, h)

    def panel(i, title):
        y = TOP + i * (PANEL_H + GAP) + (PANEL2_H - PANEL_H if i == 2 else 0)
        d.panel(MARGIN, y, W - 2 * MARGIN, PANEL2_H if i == 1 else PANEL_H, title)
        return y

    # 1. the grid of workgroups
    y0 = panel(0, f"Dispatch grid: n = {gpu.GRID_N} elements, {groups} workgroups of {wg} invocations")
    labels = [("workgroup 0", f"global 0 - {wg - 1}"), ("workgroup 1", f"global {wg} - {2 * wg - 1}"),
              ("workgroup 2", f"global {2 * wg} - {3 * wg - 1}"), None,
              (f"workgroup {last}", f"global {first_idx} - {first_idx + wg - 1}")]
    cy = y0 + 76
    top_boxes = []
    for cx, lab in zip(SLOTS, labels):
        if lab is None:
            d.text(cx, cy, ["..."], size=18, color="#4a5568")
            top_boxes.append(None)
        else:
            kind = "accent" if lab[0].endswith(str(last)) else "model"
            top_boxes.append(d.box(cx, cy, BOX_W, BOX_H, list(lab), kind))
    # 2. one workgroup
    y1 = panel(1, f"One workgroup (workgroup {last}): {wg} invocations on one compute unit")
    xa, xb = 215, 565
    d.box(W / 2, y1 + 58, 640, 32, ["shared memory: every invocation of this workgroup reads and writes it; a barrier orders them"],
          "result")
    d.box(xa, y1 + 110, 280, 40, [f"subgroup 0: invocations {first_idx} - {first_idx + sg - 1}"], "model")
    s1 = d.box(xb, y1 + 110, 280, 40, [f"subgroup 1: invocations {sub1} - {sub1 + sg - 1}"], "accent")
    d.arrow(top_boxes[-1]["b"], (top_boxes[-1]["b"][0], y1 - 1), None)
    # 3. one subgroup
    y2 = panel(2, f"One subgroup (subgroup 1): {sg} lanes execute each instruction together")
    cy3 = y2 + 66
    lanes = [(0, "lane 0"), (1, "lane 1"), (2, "lane 2"), None, (sg - 1, f"lane {sg - 1}")]
    for cx, lane in zip(SLOTS, lanes):
        if lane is None:
            d.text(cx, cy3, ["..."], size=18, color="#4a5568")
        else:
            d.box(cx, cy3, BOX_W, BOX_H, [lane[1], f"invocation {sub1 + lane[0]}"], "plain")
    d.text(W / 2, y2 + 105, ["each invocation has its own registers; in one instruction all lanes do the same operation"],
           size=12, color="#4a5568")
    d.arrow(s1["b"], (s1["b"][0], y2 - 1), None)
    return d.svg()
