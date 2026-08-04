#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Speak `text` in the cloned reference voice via the brain TTS server.

    python voice-clone.py "hi, this is my voice clone"      # -> speakers
    python voice-clone.py "..." --out clone.wav             # -> WAV
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain_tts_client import DEFAULT_SOCK, stream_request  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="voice clone via the brain TTS server")
    p.add_argument("text", help="text to speak in the cloned voice")
    p.add_argument("--socket", default=DEFAULT_SOCK)
    p.add_argument("--out", help="write a WAV instead of (only) playing")
    p.add_argument("--lang", default="english")
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=256)
    a = p.parse_args()
    stream_request(
        a.socket,
        {
            "engine": "clone",
            "text": a.text,
            "lang": a.lang,
            "temp": a.temp,
            "top_k": a.top_k,
            "seed": a.seed,
            "max_frames": a.max_frames,
        },
        out=a.out,
    )


if __name__ == "__main__":
    main()
