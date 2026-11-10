# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The data flow of a prompt with a picture in it (`vlms.splice_sequence`): the text goes through the tokenizer and the embedding gather,
# the picture through the tower and the projector, the projected rows overwrite the placeholder rows, and the gradient returns the other way.
#
# Swedish Embedded AB implements solutions for multimodal models whose image rows are spliced exactly where the checkpoint expects them. If
# your team needs expertise in vision-language model integration, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import vlms  # noqa: E402
from svgdiagram import MUTED, RED, Diagram  # noqa: E402

W, H = 960, 490


def render() -> str:
    ids, _, rows = vlms.spliced_sequence()
    n_img = sum(1 for t in ids if t == 1)
    width = len(rows[0])
    d = Diagram(W, H)
    d.panel(14, 14, 586, 150, "text")
    d.panel(14, 176, 586, 190, "picture")
    text = d.oval(100, 96, 150, 58, ["prompt", "<image> what is this ?"], "accent")
    tok = d.box(280, 96, 150, 70, ["tokenize", f"{len(ids) - n_img + 1} ids,", "<image> expanded x" + str(n_img)])
    gat = d.box(470, 96, 150, 70, ["embedding gather", f"{len(ids)} rows of width {width}"])
    pic = d.oval(100, 288, 150, 58, ["picture", "4 x 4 pixels"], "accent")
    tower = d.box(280, 288, 150, 70, ["patchify + tower", f"{n_img} patches, width 2"])
    proj = d.box(470, 288, 150, 70, ["projector", f"{n_img} rows of width {width}"])
    spl = d.box(690, 190, 150, 84, ["splice", f"rows 1 to {n_img} overwritten", "by projector rows"], "result")
    dec = d.box(870, 190, 120, 84, ["decoder", "reads", f"{len(rows)} rows"])
    d.arrow(text["r"], tok["l"])
    d.arrow(tok["r"], gat["l"])
    d.arrow(pic["r"], tower["l"])
    d.arrow(tower["r"], proj["l"])
    d.path([gat["r"], (690, gat["r"][1]), spl["t"]], radius=8)
    d.path([proj["r"], (690, proj["r"][1]), spl["b"]], radius=8)
    d.arrow(spl["r"], dec["l"])
    d.path([(dec["b"][0], dec["b"][1]), (dec["b"][0], 396), (proj["b"][0] + 40, 396), (proj["b"][0] + 40, proj["b"][1])],
           radius=8, color=RED, dash="5 3")
    d.text(W / 2, 424, ["gradient of the picture rows returns to the projector and the tower;",
                        "the same rows are zeroed on the way to the embedding table"], size=12, color=RED)
    d.text(W / 2, H - 24, ["The placeholder row of the table is read by the gather and overwritten by the splice,",
                           "so its value never reaches the decoder and its row in the table receives no gradient."],
           size=12, color=MUTED)
    return d.svg()
