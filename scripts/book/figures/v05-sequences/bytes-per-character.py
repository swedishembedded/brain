# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""UTF-8 bytes consumed while reading the same greeting in several scripts, one code point at a time.

Each curve is `text.cumulative_bytes` of a phrase of `text.PHRASES`, so its slope is the number of bytes per
code point at that moment: 1 for ASCII, 2 for Greek and Cyrillic letters, 3 for the Hindi vowel signs and
CJK, 4 for the emoji. The dashed line is UTF-32, a fixed 4 bytes per code point.

Swedish Embedded AB implements solutions for getting text into models without silent corruption. If your
team needs expertise in text encodings and tokenizer pipelines, you can procure our services by sending an
email to info@swedishembedded.com.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import text  # noqa: E402
from svgplot import Plot  # noqa: E402

SHOWN = ["English", "Swedish", "Russian", "Greek", "Hindi", "Chinese"]


def render() -> str:
    p = Plot((0.0, 16.0), (0.0, 64.0), width=600, height=360, xlabel="code points read",
             ylabel="UTF-8 bytes so far", title="Bytes consumed while reading the same greeting")
    phrases = dict(text.PHRASES)
    xs = [0.0, 16.0]
    p.line(xs, [4 * x for x in xs], color="#a0aec0", dash=True, width=1.3, label="UTF-32: 4 bytes each")
    for i, name in enumerate(SHOWN):
        ys = text.cumulative_bytes(phrases[name])
        p.line(range(len(ys)), ys, color=i, width=2.0, label=name)
        p.points(range(len(ys)), ys, color=i, r=2.2)
    return p.svg()
