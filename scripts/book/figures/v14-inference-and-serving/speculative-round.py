# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# One round of greedy speculative decoding in the chapter "Speculative Decoding": the draft's proposals, the rows the
# target scores in one pass, the target's best token at each row, the accepted prefix and the target's own token. The
# tokens are round 1 of the worked module's trace (prompt `the dog`, order-2 draft, order-4 target, k = 3).
#
# Swedish Embedded AB implements solutions for generating text from large models at a fraction of the work per token for
# its clients. If your team needs expertise in speculative decoding then you can procure our services by sending an email
# to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import speculative_decoding as sd  # noqa: E402
from svgdiagram import MUTED, RED, Diagram  # noqa: E402
import language_modeling as lm  # noqa: E402

W, H = 940, 420
COL0, STEP = 300, 170          # x of the first row's column and the distance between columns
BOX_W, BOX_H = 120, 44


def render() -> str:
    _, rounds = sd.trace_rounds()
    r = rounds[0]
    ctx, props, rows, a = r["ctx"], r["props"], r["rows"], r["accepted"]
    word = lambda t: lm.TOKENS[t]  # noqa: E731
    d = Diagram(W, H)
    d.text(110, 60, ["draft proposes"], size=13, color=MUTED)
    d.text(110, 175, ["rows of one", "target pass"], size=13, color=MUTED)
    d.text(110, 290, ["target's best", "token per row"], size=13, color=MUTED)
    inputs = [ctx[-1]] + props
    for j, tok in enumerate(inputs):
        x = COL0 + j * STEP
        if j > 0:
            prop = d.box(x, 60, BOX_W, BOX_H, [f"d{j} = {word(tok)}"], "accent")
        row = d.box(x, 175, BOX_W, BOX_H, [f"row {j}: {word(tok)}"], "model")
        if j > 0:
            d.arrow(prop["b"], row["t"])
        kind = "result" if j <= a else "plain"
        out = d.box(x, 290, BOX_W, BOX_H, [word(rows[j])], kind)
        d.arrow(row["b"], out["t"])
        if j < len(props):
            verdict = "= d%d, kept" % (j + 1) if j < a else ("differs from d%d" % (j + 1) if j == a else "not used")
            color = RED if j == a else MUTED
            d.text(x, 336, [verdict], size=12, color=color)
        else:
            d.text(x, 336, ["not used" if a < len(props) else "bonus"], size=12, color=MUTED)
    emitted = " ".join(word(t) for t in r["emit"])
    d.text(W / 2, 385, [f"context `{' '.join(word(t) for t in ctx)}`: {a} of {len(props)} proposals accepted; "
                        f"the round emits `{emitted}` for one target pass"], size=13)
    return d.svg()
