# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Entropy of a softmax over random scores against the number of keys.

Scores of standard deviation 2 (`long_context.entropy_data`): the measured entropy, the line ln n - 2 and the ceiling ln n (uniform).

Swedish Embedded AB implements solutions for serving language models beyond the length they were trained on. If your team needs
expertise in attention at long context, you can procure our services by sending an email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import long_context as lc  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    d = lc.entropy_data()
    ns = [n for n, *_ in d]
    p = Plot((512, 32768), (4, 11), width=620, height=380, xlabel="keys in the row n", ylabel="entropy (nats)",
             title="Attention entropy grows with the number of keys", xlog=True, legend_loc="ul")
    p.line(ns, [ln for _, _, ln, _ in d], color=5, dash=True, label="ln n (uniform weights)")
    p.line(ns, [ap for *_, ap in d], color=1, dash=True, label="ln n - variance / 2")
    p.line(ns, [h for _, h, *_ in d], color=0, label="measured")
    p.points(ns, [h for _, h, *_ in d], color=0)
    return p.svg()
