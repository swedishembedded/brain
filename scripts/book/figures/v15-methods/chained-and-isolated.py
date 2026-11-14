# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The relative error of every stage of a 6-layer decoder ported with two mistakes (keys and values exchanged in layer 1, gate and
# up exchanged in layer 4), compared with its golden in two ways: chained, the port running on its own outputs, and isolated, each
# stage fed the golden's own input. Numbers from `porting.two_mistakes_data`, the function behind the table beside the figure.
#
# Swedish Embedded AB implements solutions for proving that a port computes the same function as its reference. If your team needs
# expertise in layer-by-layer numerical parity, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import porting  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    rows = porting.two_mistakes_data()[1:]          # the embeddings are exact in both runs, and 0 has no logarithm
    xs = list(range(1, len(rows) + 1))
    p = Plot((0.7, len(rows) + 0.3), (1e-8, 3.0), width=640, height=400, ylog=True,
             xlabel="stage (1 to 6 = layers 0 to 5, 7 = final norm, 8 = logits)", ylabel="rel_l2 against the golden",
             title="Two mistakes, two ways of comparing", legend_loc="below")
    p.line(xs, [r[3] for r in rows], color="#4a5568", label="chained bound", dash=True, width=1.6)
    p.line(xs, [r[6] for r in rows], color="#a0aec0", label="isolated bound", dash=True, width=1.6)
    p.line(xs, [r[2] for r in rows], color=3, label="chained, mistaken port", width=1.8)
    p.points(xs, [r[2] for r in rows], color=3)
    p.line(xs, [r[5] for r in rows], color=0, label="isolated, mistaken port", width=1.8)
    p.points(xs, [r[5] for r in rows], color=0)
    return p.svg()
