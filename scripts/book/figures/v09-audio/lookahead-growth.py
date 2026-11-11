# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# How far ahead of a row (in seconds of audio) a stack of attention layers can read, for a window that slides (3 rows ahead in every layer)
# and for the chunked mask of the Nemotron encoder. The data are `speech_recognition.lookahead_data`, the dependency sets of the masks
# composed layer by layer.
#
# Swedish Embedded AB implements solutions for streaming speech recognition with a counted delay. If your team needs expertise in
# low-latency speech encoders, you can procure our services by sending an email to info@swedishembedded.com.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worked"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import speech_recognition as sr  # noqa: E402
from svgplot import Plot  # noqa: E402


def render() -> str:
    data = sr.lookahead_data()
    layers = [row[0] for row in data]
    sliding = [0.08 * row[1] for row in data]
    chunked = [0.08 * row[2] for row in data]
    p = Plot((0, 24.5), (0, 6.2), width=560, height=340, xlabel="attention layers in the stack", ylabel="audio the last layer reads ahead (seconds)",
             title="Look-ahead of a stack of masked attention layers", legend_loc="ul")
    p.line(layers, sliding, color=1, label="window sliding with 3 rows ahead", width=2.0)
    p.points(layers, sliding, color=1, r=2.5)
    p.line(layers, chunked, color=0, label="chunks of 4 rows (Nemotron)", width=2.0)
    p.points(layers, chunked, color=0, r=2.5)
    return p.svg()
