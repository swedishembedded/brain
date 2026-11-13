# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The path of one request through a small decision model: the state encoder (a sentence vector from the text, scaled numbers from the
# fields), the decision head, the temperature, the decision rule that prices the probabilities, and the record written beside the action.
#
# Swedish Embedded AB implements solutions for decisions that must be fast and explainable. If your team needs expertise in decision
# models inside applications, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import Diagram, text_width  # noqa: E402


def render() -> str:
    pad, gap, h = 20, 34, 74
    col1 = [["request text", "five words"], ["request fields", "amount, age,", "device, contacts"]]
    col2 = [["sentence encoder", "mean of word vectors,", "length 1: 6 numbers"], ["scaling", "(v - mean) / sd:", "4 numbers"]]
    col3 = [["state vector x", "10 numbers"]]
    col4 = [["decision head", "z = W x + b", "4 logits"]]
    col5 = [["temperature", "p = softmax(z / T)"]]
    col6 = [["decision rule", "act if p_max >= 1 - c_e / c_w", "otherwise escalate"]]
    cols = [col1, col2, col3, col4, col5, col6]
    widths = [int(max(text_width(b) for b in c) + 24) for c in cols]
    width = pad * 2 + sum(widths) + gap * (len(cols) - 1)
    ytop, ybot = pad + 50 + h / 2, pad + 50 + h / 2 + h + 40
    ymid = (ytop + ybot) / 2
    rec_y = ybot + h / 2 + 52
    d = Diagram(int(width), int(rec_y + 28 + pad))
    xs, x = [], pad
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    b1 = d.box(xs[0], ytop, widths[0], h, col1[0], "plain")
    b2 = d.box(xs[0], ybot, widths[0], h, col1[1], "plain")
    e1 = d.box(xs[1], ytop, widths[1], h, col2[0], "model")
    e2 = d.box(xs[1], ybot, widths[1], h, col2[1], "model")
    sv = d.box(xs[2], ymid, widths[2], h, col3[0], "result")
    hd = d.box(xs[3], ymid, widths[3], h, col4[0], "accent")
    tp = d.box(xs[4], ymid, widths[4], h, col5[0], "model")
    dr = d.box(xs[5], ymid, widths[5], h, col6[0], "accent")
    d.arrow(b1["r"], e1["l"])
    d.arrow(b2["r"], e2["l"])
    d.path([e1["r"], (e1["r"][0] + gap / 2, ytop), (e1["r"][0] + gap / 2, ymid - 14), (sv["l"][0], ymid - 14)])
    d.path([e2["r"], (e2["r"][0] + gap / 2, ybot), (e2["r"][0] + gap / 2, ymid + 14), (sv["l"][0], ymid + 14)])
    d.arrow(sv["r"], hd["l"])
    d.arrow(hd["r"], tp["l"])
    d.arrow(tp["r"], dr["l"])
    rec = d.box((xs[2] + xs[5]) / 2, rec_y, xs[5] - xs[2] + widths[5] / 2 + 40, 44,
                ["record: state vector, logits, T, probabilities, threshold, action, model identity"], "plain")
    d.line((xs[2], ymid + h / 2), (xs[2], rec_y - 22))
    d.line((xs[3], ymid + h / 2), (xs[3], rec_y - 22))
    d.arrow((xs[5], ymid + h / 2), (xs[5], rec_y - 22))
    d.text(xs[5] - 16 - 62, (ymid + h / 2 + rec_y - 22) / 2 - 8, ["action: route to a team", "or escalate"], size=12, color="#4a5568")
    return d.svg()
