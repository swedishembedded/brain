# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# An encoder-decoder pair on the input `2 3 1` of the digit-reversal task: the encoder reads the input into one state, the
# context vector, and the decoder writes `1 3 2 <end>` from that vector alone, each step fed the token before it (the target
# shifted right when training, its own previous output when generating).
#
# Swedish Embedded AB implements solutions for models that turn one sequence into another. If your team needs expertise in
# sequence-to-sequence models, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram  # noqa: E402

W, H = 1000, 400
SRC = ["2", "3", "1"]
OUT = SRC[::-1] + ["<end>"]
FED = ["<s>"] + SRC[::-1]


def render() -> str:
    d = Diagram(W, H)
    d.panel(10, 10, 360, 340, "Encoder: reads the input")
    d.panel(570, 10, 420, 340, "Decoder: writes the output")
    y = 190
    enc_x = [85, 195, 305]
    dec_x = [625, 725, 825, 925]
    enc = [d.box(x, y, 84, 50, [f"GRU {i + 1}"], "model") for i, x in enumerate(enc_x)]
    dec = [d.box(x, y, 80, 50, [f"GRU {i + 1}"], "model") for i, x in enumerate(dec_x)]
    ctx = d.oval(470, y, 124, 76, ["context vector", "c = h3", "H numbers"], "result")
    for i, x in enumerate(enc_x):
        t = d.box(x, 300, 74, 34, [f"x{i + 1} = {SRC[i]}"], "accent")
        d.arrow(t["t"], enc[i]["b"])
    for i in range(2):
        d.arrow(enc[i]["r"], enc[i + 1]["l"], f"h{i + 1}", side="above")
    d.arrow(enc[2]["r"], ctx["l"], "h3", side="above")
    d.arrow(ctx["r"], dec[0]["l"], "s0 = c", side="above")
    for i in range(3):
        d.arrow(dec[i]["r"], dec[i + 1]["l"], f"s{i + 1}", side="above")
    for i, x in enumerate(dec_x):
        t = d.box(x, 300, 74, 34, [f"in {FED[i]}"], "accent")
        d.arrow(t["t"], dec[i]["b"])
        o = d.oval(x, 90, 74, 36, [f"{OUT[i]}"], "result")
        d.arrow(dec[i]["t"], o["b"])
    d.text(W / 2, 376, ["the decoder sees the input only through c; in training each input token is the target shifted right, "
                        "when generating it is the model's own previous output"], size=12, color=MUTED)
    return d.svg()
