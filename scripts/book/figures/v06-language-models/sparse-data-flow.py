# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The two ways a query picks its keys and the one attention that reads them. Top: the key-level indexer (a cheap score for every
# cached key, then the top k). Middle: the bucket-level index (a score for every bucket centroid, then the members of the best
# buckets, plus the window and the sinks). Bottom: the selection, in either case a list of key positions per query, is resolved
# through the block table of the paged cache and attended exactly. Widths are those of the chapter's examples.
#
# Swedish Embedded AB implements solutions for serving long-context language models whose attention cost is set by what a token
# needs. If your team needs expertise in sparse attention, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 980, 570
PW = 470


def chain(d, x0, title, q_lines, k_lines, score_lines, pick_lines, sel_lines, score_label, pick_label):
    """One indexer panel: two inputs on top, then score, pick and the selection, stacked and centred on the panel."""
    d.panel(x0, 10, PW, 290, title)
    cx = x0 + PW / 2
    q = d.oval(cx - 110, 80, 170, 44, q_lines, "accent")
    k = d.oval(cx + 110, 80, 170, 44, k_lines, "result")
    score = d.box(cx, 150, 230, 50, score_lines)
    pick = d.box(cx, 215, 230, 50, pick_lines)
    sel = d.oval(cx, 275, 200, 40, sel_lines, "result")
    d.arrow(q["b"], (cx - 60, score["t"][1]))
    d.arrow(k["b"], (cx + 60, score["t"][1]))
    d.arrow(score["b"], pick["t"], label=score_label, side="right")
    d.arrow(pick["b"], sel["t"], label=pick_label, side="right")
    return sel


def render() -> str:
    d = Diagram(W, H)
    sel1 = chain(d, 10, "Key-level indexer: a score for every cached key", ["query token:", "indexer query"], ["indexer keys,", "T rows"],
                 ["score every key", "T cheap dot products"], ["keep the k", "largest scores"], ["selection: k positions"], "T scores", "")
    sel2 = chain(d, 500, "Bucket-level index: a score for every bucket", ["query token:", "index query"], ["bucket", "centroids"],
                 ["score every bucket", "about T / 8 dot products"], ["members of the best", "buckets, window, sinks"], ["selection: some thousand keys"],
                 "bucket scores", "")
    d.panel(10, 315, 960, 245, "Attention over the selection")
    blk = d.box(490, 385, 280, 50, ["block table", "position -> (page, row)"])
    att = d.box(490, 460, 280, 50, ["attention over the selection", "one softmax over its scores"])
    out = d.oval(490, 530, 200, 40, ["output, as wide as dense"], "accent")
    d.path([sel1["b"], (sel1["b"][0], 385), blk["l"]])
    d.path([sel2["b"], (sel2["b"][0], 385), blk["r"]])
    d.arrow(blk["b"], att["t"], label="K, V rows", side="right")
    d.arrow(att["b"], out["t"])
    d.text(237, 362, ["key positions per query"], size=12, color=MUTED, anchor="end")
    d.text(743, 362, ["key positions per query"], size=12, color=MUTED, anchor="start")
    return d.svg()
