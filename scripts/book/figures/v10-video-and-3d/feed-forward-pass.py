# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One forward pass of brain's feed-forward reconstruction model (`crates/worldmirror2`): photographs, an optional set of known
# cameras, the patch encoder, the token assembly, the alternating trunk with its four taps, the five heads, and the assembly of a
# scene. The token counts come from the worked module's configuration constants.
#
# Swedish Embedded AB implements feed-forward 3D reconstruction pipelines that accept whatever a capture rig already knows. If your
# team needs expertise in learned multi-view geometry, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import feed_forward_3d as ff  # noqa: E402
from svgdiagram import Diagram, text_width  # noqa: E402


def render() -> str:
    grid = ff.IMG // ff.PATCH
    patches = grid * grid
    per_frame = ff.SPECIALS + patches
    cols = {
        "photos": ["S photographs", "no poses,", "no calibration"],
        "known": ["known cameras", "(optional)"],
        "enc": ["patch encoder", f"{ff.DEPTH} blocks, one", "frame at a time", f"{patches:,} tokens"],
        "tok": ["token assembly", "camera, 4 registers,", "pose, intrinsics,", f"patches: {per_frame:,}", "tokens a frame"],
        "trunk": [f"trunk, {ff.DEPTH} times:", "frame block,", "global block", "taps after levels", ", ".join(str(t) for t in ff.TAPS)],
        "asm": ["assembly", "fuse the views,", "filter pixels"],
        "out": ["gaussian scene", "and S cameras"],
    }
    heads = [["camera head", "9 numbers a frame"], ["depth head", "depth, confidence, mask"], ["pointmap head", "xyz, confidence"],
             ["normal head", "normal, confidence"], ["gaussian head", "depth, mask, 12 values"]]
    w = {k: int(text_width(v) + 28) for k, v in cols.items()}
    hw = int(max(text_width(h) for h in heads) + 28)
    gap = 36
    h_box = 64
    head_gap = 14
    n_h = len(heads)
    heads_h = n_h * h_box + (n_h - 1) * head_gap
    pad = 24
    mid_y = pad + heads_h / 2
    x = pad
    pos = {}
    for k in ("photos", "enc", "tok", "trunk"):
        pos[k] = x + w[k] / 2
        x += w[k] + gap
    pos["heads"] = x + hw / 2
    x += hw + gap
    for k in ("asm", "out"):
        pos[k] = x + w[k] / 2
        x += w[k] + gap
    width = int(x - gap + pad)
    known_y = mid_y + 74
    height = int(pad + heads_h + pad)
    d = Diagram(width, height)
    tall = {"enc": 88, "tok": 100, "trunk": 112, "photos": 74, "asm": 74, "out": 62}
    box = {}
    for k in ("photos", "enc", "tok", "trunk", "asm"):
        kind = "result" if k == "asm" else "model"
        box[k] = d.box(pos[k], mid_y, w[k], tall[k], cols[k], kind)
    box["out"] = d.box(pos["out"], mid_y, w["out"], tall["out"], cols["out"], "result")
    box["known"] = d.box(pos["photos"], known_y, w["photos"], 50, cols["known"], "plain")
    hb = []
    for i, lines in enumerate(heads):
        cy = pad + h_box / 2 + i * (h_box + head_gap)
        hb.append(d.box(pos["heads"], cy, hw, h_box, lines, "accent"))
    for a, b in (("photos", "enc"), ("enc", "tok"), ("tok", "trunk")):
        d.arrow(box[a]["r"], box[b]["l"])
    # the known cameras enter the token assembly by the pose and intrinsics tokens
    d.path([box["known"]["r"], (pos["tok"], known_y), box["tok"]["b"]], dash="5 3")
    for i, hbox in enumerate(hb):
        off = (i - (n_h - 1) / 2) * 12
        tr_r = (box["trunk"]["r"][0], mid_y + off)
        as_l = (box["asm"]["l"][0], mid_y + off)
        d.arrow(tr_r, hbox["l"])
        d.arrow(hbox["r"], as_l)
    d.arrow(box["asm"]["r"], box["out"]["l"])
    return d.svg()
