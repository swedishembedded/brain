# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One cross-attention site with an identity branch: the image tokens' queries attend the text rows and, separately, the identity
# tokens (each with its own keys and values), and the two contexts are added, the second scaled, before the site's shared output
# projection. The widths are those of InstantID on SDXL (77 text rows of 2048, 16 identity tokens of 2048).
#
# Swedish Embedded AB implements solutions for conditioning image generators on a person's identity with the injection points
# gated one by one. If your team needs expertise in identity adapters, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram  # noqa: E402

W, H = 940, 500
TOP, MID, BOT = 90, 250, 410
ARROW = "#2b6cb0"


def render() -> str:
    d = Diagram(W, H)
    d.colors.add(ARROW)
    text = d.oval(70, TOP, 120, 44, ["text rows", "77 x 2048"], "plain")
    kv = d.box(250, TOP, 150, 50, ["site keys and", "values (text)"], "model")
    att = d.box(470, TOP, 150, 54, ["attention over", "the text rows"], "model")
    q = d.oval(470, MID, 170, 44, ["queries of the", "image tokens"], "plain")
    emb = d.oval(70, BOT, 120, 44, ["identity", "embedding"], "plain")
    res = d.box(210, BOT, 100, 50, ["resampler"], "accent")
    tok = d.oval(330, BOT, 100, 44, ["ID tokens", "16 x 2048"], "plain")
    kvi = d.box(480, BOT - 0, 130, 50, ["this site's own", "to_k_ip, to_v_ip"], "accent")
    atti = d.box(640, BOT, 140, 54, ["attention over", "the ID tokens"], "accent")
    ctx = d.oval(640, MID, 150, 54, ["context =", "text + s x ID"], "result")
    outp = d.box(810, MID, 100, 50, ["shared", "to_out"], "model")
    add = d.oval(810, 130, 150, 44, ["added to the", "image tokens"], "plain")

    d.arrow(text["r"], kv["l"])
    d.arrow(kv["r"], att["l"])
    d.arrow(q["t"], att["b"])
    d.path([q["b"], (470, 335), (600, 335), (600, BOT - 27)])
    d.text(535, 321, ["the same queries"], size=12, color="#4a5568")
    d.arrow(emb["r"], res["l"], color=ARROW)
    d.arrow(res["r"], tok["l"], color=ARROW)
    d.arrow(tok["r"], kvi["l"], color=ARROW)
    d.arrow(kvi["r"], atti["l"], color=ARROW)
    d.arrow((680, BOT - 27), (680, 274), "scale s", side="right", color=ARROW)
    d.path([att["r"], (640, TOP), ctx["t"]])
    d.arrow(ctx["r"], outp["l"])
    d.arrow(outp["t"], add["b"])
    d.text(W / 2, 470, ["blue: the identity branch, with its own projections at every site (70 sites on SDXL). The text attention is unchanged;",
                        "its softmax never sees the identity tokens, so with s = 0 the site's output is exactly the text-only output."],
           size=12, color="#4a5568")
    return d.svg()
