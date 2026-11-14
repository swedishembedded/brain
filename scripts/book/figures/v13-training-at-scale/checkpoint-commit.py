# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The commit of one generation in the chapter "Checkpoints, Determinism and Recovery": the files of a step are written into a temporary
# directory and fsynced, the meta file with the length and digest of the others last, the directory is renamed to its generation name,
# and only the rename publishes it. A crash before the rename leaves the older generations the newest; a reader takes the newest
# generation whose digests verify.
#
# Swedish Embedded AB implements solutions for crash-safe, resumable training runs for its clients. If your team needs expertise in
# checkpointing and exact resume of large training jobs, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from svgdiagram import INK, MUTED, RED, Diagram  # noqa: E402

W, H = 1000, 340
XS = (110, 340, 570, 800)
Y1, Y2 = 70, 250


def render() -> str:
    d = Diagram(W, H)
    # the writer's steps, left to right; only the last one publishes
    values = d.box(XS[0], Y1, 170, 60, ["values.bin", "write, fsync"], "model")
    state = d.box(XS[1], Y1, 170, 60, ["state.bin", "write, fsync"], "model")
    meta = d.box(XS[2], Y1, 170, 60, ["meta.json, last", "lengths, digests, step"], "accent")
    rename = d.box(XS[3], Y1, 170, 60, ["rename to gen-N", "fsync of the root"], "result")
    d.arrow(values["r"], state["l"])
    d.arrow(state["r"], meta["l"])
    d.arrow(meta["r"], rename["l"], "commit", above=True, color=RED)
    d.text((XS[0] + XS[2]) / 2, Y1 - 52, ["written inside tmp-N-pid, a name no reader looks for"], size=12, color=MUTED)
    # the reader's side
    pick = d.oval(XS[1], Y2, 170, 60, ["restore", "newest that verifies"], "result")
    old = d.box(XS[2], Y2, 170, 60, ["gen-(N-1)", "kept, digests recomputed"], "model")
    new = d.box(XS[3], Y2, 170, 60, ["gen-N", "digests recomputed"], "accent")
    d.arrow(rename["b"], new["t"], "publishes", side="right")
    d.arrow(old["l"], pick["r"], "second", above=True)
    d.path([new["b"], (XS[3], Y2 + 62), (XS[1], Y2 + 62), (pick["b"][0], pick["b"][1])], color=INK)
    d.text((XS[2] + XS[3]) / 2, Y2 + 78, ["first: gen-N, and gen-(N-1) when gen-N fails its digests"], size=12, color=MUTED)
    d.text(XS[0], Y2, ["a crash before", "the rename leaves", "these as they were"], size=12, color=MUTED)
    return d.svg()
