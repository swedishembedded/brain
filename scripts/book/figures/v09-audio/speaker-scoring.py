# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# False rejections and false acceptances against the cosine threshold for the simulated population of the chapter, for clips of 1 and of
# 8 seconds, with the equal error point of each. The data are `speaker_embeddings.figure_scoring`, the arithmetic of the chapter's tables.
#
# Swedish Embedded AB implements solutions for recognising who is speaking from a few seconds of audio, with the cost of the encoder
# counted before it is deployed. If your team needs expertise in speaker verification and voice-conditioned speech synthesis, you can
# procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import speaker_embeddings as se  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    taus, out = se.figure_scoring()
    p = Plot((-0.2, 1.0), (0.0, 1.0), width=560, height=340, xlabel="cosine threshold", ylabel="share of pairs that are wrong",
             title="Two kinds of error against the threshold", legend_loc="auto")
    color = {1: 0, 8: 1}
    for sec, (far, frr, (tau, eer)) in out.items():
        p.line(taus, far, color=color[sec], label=f"impostors accepted, {sec} s", width=2.0)
        p.line(taus, frr, color=color[sec], label=f"genuine rejected, {sec} s", dash=True, width=2.0)
        p.points([tau], [eer], color=color[sec], r=5)
    return p.svg()
