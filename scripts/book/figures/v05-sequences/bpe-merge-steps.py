# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The nine words of the hand-trained corpus as rows of tokens, shown at six moments of BPE training: before any
# merge and after merges 1, 2, 4, 6 and 10. A token made by the merge of that step is shaded. The states come
# from `tokenization.merge_states`, the chapter's own training loop, so the picture is the table's merge list.
#
# Swedish Embedded AB implements solutions for building vocabularies that fit a corpus. If your team needs
# expertise in tokenizer training, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tokenization  # noqa: E402
from svgdiagram import Diagram, text_width  # noqa: E402

STEPS = (0, 1, 2, 4, 6, 10)
BOX_H, ROW_H, GAP, PAD = 22, 30, 3, 5
LABEL_W, COL_GAP, TOP = 70, 22, 56


def box_w(token):
    return max(20, text_width(token) + 2 * PAD)


def render() -> str:
    words, counts, states, made = tokenization.merge_states(STEPS)
    cols = []
    for k in STEPS:
        widest = max(sum(box_w(t) for t in w) + GAP * (len(w) - 1) for w in states[k])
        cols.append(widest)
    width = int(LABEL_W + sum(cols) + COL_GAP * len(cols) + 18)
    height = TOP + ROW_H * len(words) + 18
    d = Diagram(width, height)
    x = LABEL_W + COL_GAP // 2
    starts = []
    for k, w in zip(STEPS, cols):
        starts.append(x)
        title = ["start"] if k == 0 else [f"merge {k}", f"makes {made[k]}"]
        d.text(x + w / 2, 22, title, size=12, color="#1a202c", weight="bold")
        x += w + COL_GAP
    for r, (word, count) in enumerate(zip(words, counts)):
        cy = TOP + ROW_H * r + ROW_H / 2
        d.text(LABEL_W - 8, cy, [f"{word} x{count}"], size=12, anchor="end")
        for k, x0 in zip(STEPS, starts):
            x = x0
            prev = states[STEPS[STEPS.index(k) - 1]][r] if k else []
            for t in states[k][r]:
                w = box_w(t)
                new = k > 0 and t not in prev and t == made.get(k)
                d.box(x + w / 2, cy, w, BOX_H, [t], "result" if t == made.get(k) and k > 0 else "plain")
                x += w + GAP
    return d.svg()
