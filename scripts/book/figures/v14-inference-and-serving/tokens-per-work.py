# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Tokens per unit of work of speculative rounds against the number of proposals k, in the chapter "Speculative
# Decoding": a small draft decoder of a tenth of the target's work per pass, at one sequence and at 32 sequences, for
# three per-position acceptances. The data are the worked module's `tokens_per_unit`, so the curves cannot disagree with
# the chapter's table.
#
# Swedish Embedded AB implements solutions for generating text from large models at a fraction of the work per token for
# its clients. If your team needs expertise in speculative decoding then you can procure our services by sending an email
# to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import speculative_decoding as sd  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    ks = list(range(1, 11))
    p = Plot((1, 10), (0, 5.2), width=640, height=420, xlabel="proposals per round, k",
             ylabel="tokens per unit of work", title="A small draft decoder: tokens per plain step of work", legend_loc="ul")
    for color, a in enumerate(sd.ALPHAS):
        p.line(ks, [sd.tokens_per_unit("small decoder", 1, a, k) for k in ks], color=color, label=f"alpha {a}, 1 sequence")
        p.line(ks, [sd.tokens_per_unit("small decoder", 32, a, k) for k in ks], color=color, dash=True,
               label=f"alpha {a}, 32 sequences")
    p.hline(1.0)
    return p.svg()
