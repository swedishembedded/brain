# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The path of the capstone: raw text, cleaning and the split, the tokenizer, the base run and the adapter run (each killed once and
# resumed from its generations), the four files that cross to serving, the loader that checks them, the server under continuous
# batching, and the reference whose answers every served answer is compared with. The numbers are the worked module's constants.
#
# Swedish Embedded AB implements solutions for machine learning systems that are verified from the raw data to the served answer.
# If your team needs expertise in end-to-end training and serving pipelines then you can procure our services by sending an email
# to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import capstone as cs  # noqa: E402
from svgdiagram import INK, MUTED, Diagram  # noqa: E402

W, H = 1000, 560


def render() -> str:
    d = Diagram(W, H)
    raw = d.box(100, 70, 160, 64, [f"{cs.N_RAW} raw documents", "dirty case, spaces,", "duplicates"], "plain")
    clean = d.box(290, 70, 160, 64, ["clean, drop copies,", "split by digest"], "model")
    tokz = d.box(480, 70, 160, 64, ["learn the tokenizer", "on training text"], "model")
    base = d.box(670, 70, 160, 64, [f"base run, {cs.PRE_STEPS}", "updates, killed", f"after {cs.PRE_KILL}, resumed"], "model")
    adap = d.box(860, 70, 160, 64, [f"adapter run, {cs.FT_STEPS}", "updates, killed", f"after {cs.FT_KILL}, resumed"], "model")
    for a, b in ((raw, clean), (clean, tokz), (tokz, base), (base, adap)):
        d.arrow(a["r"], b["l"])
    man = d.note(290, 215, 160, 64, ["manifest.json:", "lengths, digests,", "trained_on"], "result")
    ftok = d.oval(480, 215, 150, 44, ["tokenizer.bin"], "result")
    fbase = d.oval(670, 215, 150, 44, ["base.bin"], "result")
    fad = d.oval(860, 215, 150, 44, ["adapter.bin"], "result")
    for a, b in ((tokz, ftok), (base, fbase), (adap, fad)):
        d.arrow(a["b"], b["t"])
    load = d.box(290, 360, 160, 64, ["loader: check,", "rebuild, merge"], "accent")
    serve = d.box(570, 360, 170, 64, [f"server: {cs.SLOTS} slots,", "continuous batching,", "cached forward"], "accent")
    ans = d.box(860, 360, 160, 64, ["8 answers"], "result")
    d.arrow(man["b"], load["t"])
    for f in (ftok, fbase, fad):
        d.arrow((f["b"][0], f["b"][1]), (load["t"][0] + 40, load["t"][1]))
    d.arrow(load["r"], serve["l"])
    d.arrow(serve["r"], ans["l"])
    req = d.oval(570, 470, 210, 44, ["8 requests, arriving at", "iterations 0 to 7"], "plain")
    d.arrow(req["t"], serve["b"])
    ref = d.box(100, 470, 160, 64, ["reference: straight", "runs, answers", "computed offline"], "plain")
    d.path([ref["b"], (100, 535), (860, 535), ans["b"]], color=MUTED, dash="5 3")
    d.text(330, 522, ["compared answer by answer, token by token"], size=12, color=MUTED)
    d.text(100, 300, ["training writes", "only these files;", "serving reads", "only these files"], size=12, color=INK)
    return d.svg()
