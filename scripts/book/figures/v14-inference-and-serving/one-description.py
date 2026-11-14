# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The surfaces of the chapter "Capabilities, Providers and APIs" and the one description they all read: callers on the
# serving machine are checked against an action's full spec, every other caller against its served view first, and served
# calls go through the one executor before they reach the model's provider.
#
# Swedish Embedded AB implements solutions for presenting many machine learning models through a few stable interfaces for
# its clients. If your team needs expertise in inference APIs and serving contracts then you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import MUTED, Diagram, text_width  # noqa: E402

W, H = 1000, 470


def _w(lines, pad=24):
    return text_width(lines) + pad


def render() -> str:
    d = Diagram(W, H)
    d.text(140, 22, ["callers"], size=13, color=MUTED, weight="bold")
    d.text(445, 22, ["the action's own description"], size=13, color=MUTED, weight="bold")
    d.text(895, 22, ["one catalog entry"], size=13, color=MUTED, weight="bold")
    left = [
        (70, ["command line: brain <arch> <action>,", "and the per-model samples"], "plain"),
        (150, ["event socket: JSON lines,", "owner-only"], "plain"),
        (250, ["HTTP /v1/run and /v1/jobs"], "plain"),
        (330, ["D-Bus: Manifests, Run, Subscribe"], "plain"),
        (410, ["HTTP dialects: chat, embeddings,", "images, speech, transcription"], "plain"),
    ]
    lw = max(_w(t) for _, t, _ in left)
    boxes = [d.box(140, y, lw, 56, t, k) for y, t, k in left]
    local = d.box(445, 110, _w(["validate:", "the full spec,", "host answers filled"]), 76,
                  ["validate:", "the full spec,", "host answers filled"], "accent")
    served = d.box(445, 290, _w(["validate_served:", "the served view first,", "then the full spec"]), 80,
                   ["validate_served:", "the served view first,", "then the full spec"], "accent")
    ex_lines = ["executor: admission,", "residency, continuous", "batching across surfaces"]
    ex = d.box(690, 290, _w(ex_lines), 80, ex_lines, "result")
    prov_lines = ["provider of the model:", "manifest, actions,", "run(invocation)"]
    pw = _w(prov_lines)
    prov = d.box(895, 200, pw, 300, prov_lines, "model")
    for b in boxes[:2]:
        d.arrow(b["r"], local["l"])
    for b in boxes[2:4]:
        d.arrow(b["r"], served["l"])
    d.path([boxes[4]["r"], (690, 410), ex["b"]])
    d.text(470, 395, ["an invocation built from the dialect's fixed fields"], size=12, color=MUTED)
    d.arrow(local["r"], (prov["l"][0], local["r"][1]))
    d.arrow(served["r"], ex["l"])
    d.arrow(ex["r"], (prov["l"][0], ex["r"][1]))
    d.text(500, 455, ["the SDK calls a model crate's typed pipeline in-process, beside this layer, not through it"],
           size=12, color=MUTED)
    return d.svg()
