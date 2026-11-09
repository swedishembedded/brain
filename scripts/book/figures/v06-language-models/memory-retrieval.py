# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Retrieval accuracy of a fixed-size key-value memory (d_k = d_v = 16) as more keys are written. Solid lines: each key is written
# three times with fresh values and the latest must come back; dashed lines: each key is written once. Numbers from
# `linear_attention.memory_accuracy_data`, the function behind the table of the chapter.
#
# Swedish Embedded AB implements solutions for serving long-context language models within a memory budget. If your team needs
# expertise in associative memory layers, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import linear_attention as la  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = la.memory_accuracy_data()
    ms = list(la.MEM["ms"])
    p = Plot((0, max(ms) + 1), (0, 1.05), width=640, height=380, xlabel="keys written to the memory (state of 16 x 16 numbers)",
             ylabel="fraction of keys retrieved correctly", title="Retrieval from a matrix memory")
    styles = (("additive", 3), ("delta, beta = 1", 1), ("delta, beta = 0.5", 0))
    for label, color in styles:
        p.line(ms, data["rewrite"][label], color=color, label=f"{label}, keys rewritten", width=2.0)
        p.points(ms, data["rewrite"][label], color=color, r=3.0)
    for label, color in styles[:2]:
        p.line(ms, data["once"][label], color=color, label=f"{label}, written once", dash=True, width=1.6)
    return p.svg()
