#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Design a voice from a natural-language instruction (VoiceDesign), or use a
CustomVoice preset speaker, via the brain TTS server.

    python voice-design.py --instruct "a deep cinematic narrator" --text "in a world..."
    python voice-design.py --engine customvoice --speaker serena --text "hi there"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain_tts_client import DEFAULT_SOCK, stream_request  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="voice design / custom voice via the brain TTS server")
    p.add_argument("--text", required=True, help="text to speak")
    p.add_argument("--instruct", default="", help="natural-language voice/style description")
    p.add_argument("--engine", default="design", choices=["design", "customvoice"])
    p.add_argument("--speaker", help="preset speaker (customvoice): serena, ryan, vivian, eric, ...")
    p.add_argument("--socket", default=DEFAULT_SOCK)
    p.add_argument("--out", help="write a WAV instead of (only) playing")
    p.add_argument("--lang", default="english")
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=256)
    a = p.parse_args()
    req = {
        "engine": a.engine,
        "text": a.text,
        "instruct": a.instruct,
        "lang": a.lang,
        "temp": a.temp,
        "top_k": a.top_k,
        "seed": a.seed,
        "max_frames": a.max_frames,
    }
    if a.speaker:
        req["speaker"] = a.speaker
    stream_request(a.socket, req, out=a.out)


if __name__ == "__main__":
    main()
