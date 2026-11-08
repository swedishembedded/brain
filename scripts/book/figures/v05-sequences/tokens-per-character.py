# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Tokens per character against the number of merges for a byte-level tokenizer trained on English text only,
# on the greetings of the previous chapter and on held-out English text. The merges are spent on English: its curve falls, the others stay at
# their bytes per character. Data from `tokenization.script_curves`.
#
# Swedish Embedded AB implements solutions for tokenizers that treat every language fairly. If your team needs
# expertise in multilingual vocabularies, you can procure our services by sending an email to
# info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tokenization  # noqa: E402
from svgplot import Plot  # noqa: E402

SHOWN = ["English, held-out text", "Swedish", "Russian", "Hindi", "Chinese", "Emoji"]


def render() -> str:
    curves = tokenization.script_curves()
    p = Plot((0.0, 70.0), (0.0, 4.4), width=600, height=340, xlabel="merges learned from English text",
             ylabel="tokens per character", title="Merges learned from English help only English")
    for i, name in enumerate(SHOWN):
        p.line(range(71), curves[name], color=i, label=name)
    return p.svg()
