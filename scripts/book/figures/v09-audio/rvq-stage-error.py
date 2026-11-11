# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The mean squared error of the trained residual quantizer of the chapter after each stage, on the training vectors and on held-out
# vectors, with the one flat codebook of 64 centres beside the two-stage point. The data are `audio_codecs.rvq_curve`, the arithmetic
# of the chapter's tables.
#
# Swedish Embedded AB implements solutions for turning audio into discrete tokens with a counted bit budget. If your team needs
# expertise in residual vector quantization, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import audio_codecs as ac  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    bits, train, held, (fbits, ftrain, fheld) = ac.rvq_curve()
    p = Plot((0, 12.5), (0.05, 2.0), width=560, height=340, xlabel="bits sent per vector (3 per stage)",
             ylabel="mean squared error (log scale)", title="Error after each stage of a residual quantizer", legend_loc="ur",
             ylog=True)
    p.line(bits, train, color=0, label="training vectors", width=2.0)
    p.points(bits, train, color=0, r=3)
    p.line(bits, held, color=1, label="held-out vectors", width=2.0)
    p.points(bits, held, color=1, r=3)
    p.points([fbits], [ftrain], color=2, label="one flat codebook of 64, training", r=5)
    p.points([fbits], [fheld], color=3, label="one flat codebook of 64, held out", r=5)
    return p.svg()
