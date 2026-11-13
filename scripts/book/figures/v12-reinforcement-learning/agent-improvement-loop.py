# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The loop of the project: tasks at a level are given to a policy, its answers are verified, the group-relative update is taken, and the
# policy is scored on a probe bank that the training never reads. The scores raise the level and, at a snapshot, decide whether the new
# version replaces the one being served. The three planted mistakes are marked where they enter.
#
# Swedish Embedded AB implements solutions for agents that improve from verified feedback and for deciding, with a stated test, which
# version may be served. If your team needs expertise in reinforcement learning from verifiable rewards, you can procure our services by
# sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import RED, Diagram, text_width  # noqa: E402


def render() -> str:
    top = [
        (["task generator:", "24 pool inputs,", "a level from 1 to 5"], "plain"),
        (["policy: 240 numbers,", "8 answers for each", "of 4 tasks per step"], "model"),
        (["exact verifier:", "reward 1 for an", "answer that is right,", "else 0"], "accent"),
        (["group advantages,", "clipped update,", "2 epochs of the", "same batch"], "model"),
    ]
    bottom = [
        (["curriculum: raise the", "level when the probe", "success at it is", "at least 0.6"], "accent"),
        (["probe bank: 48 inputs", "at each of 5 levels,", "never trained on"], "result"),
        (["gate: paired lower", "bounds against the", "served version"], "accent"),
        (["registry: promote,", "audit bank, and", "rollback to the", "previous version"], "result"),
    ]
    gap, pad, h = 100, 24, 92
    widths = [int(max(text_width(top[i][0]), text_width(bottom[i][0])) + 24) for i in range(4)]
    xs, x = [], pad
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    width = int(x - gap + pad)
    y_top, y_bot = pad + h / 2, pad + h + 84 + h / 2
    d = Diagram(width, int(y_bot + h / 2 + pad + 40))
    tb = [d.box(xs[i], y_top, widths[i], h, top[i][0], top[i][1]) for i in range(4)]
    bb = [d.box(xs[i], y_bot, widths[i], h, bottom[i][0], bottom[i][1]) for i in range(4)]
    for i in range(3):
        d.arrow(tb[i]["r"], tb[i + 1]["l"])
    d.arrow(tb[1]["b"], bb[1]["t"], label="the policy", side="right")
    d.arrow(bb[1]["l"], bb[0]["r"], label="every step")
    d.arrow(bb[0]["t"], tb[0]["b"], label="level + 1", side="right")
    d.arrow(bb[1]["r"], bb[2]["l"], label="every 10 steps")
    d.arrow(bb[2]["r"], bb[3]["l"], label="promote")
    d.text(xs[2], y_top + h / 2 + 18, ["mistake 1: a weak verifier"], size=12, color=RED)
    d.text(xs[0], y_bot + h / 2 + 24, ["mistake 2: the curriculum", "reads the training reward"], size=12, color=RED)
    d.text(xs[2], y_bot + h / 2 + 18, ["mistake 3: unpaired standard error"], size=12, color=RED)
    return d.svg()
