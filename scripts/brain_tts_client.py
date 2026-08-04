#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Client for the brain TTS server (`brain tts serve`).

Connects to the Unix socket, sends one JSONL request, and streams back the
generated 24 kHz f32 PCM `audio_chunk`s — playing them straight to the speakers
via sounddevice as they arrive (or saving to a WAV with --out).
"""
import base64
import json
import os
import socket
import sys
import tempfile

import numpy as np

# Matches the Rust server's own default exactly: crates/cli/src/tts_serve.rs
# binds std::env::temp_dir().join("brain-tts.sock") unless --socket overrides
# it, so this needs to resolve the same OS temp dir, not a literal "/tmp".
# BRAIN_TTS_SOCK / --socket (on the two callers) both override it.
DEFAULT_SOCK = os.environ.get("BRAIN_TTS_SOCK") or os.path.join(tempfile.gettempdir(), "brain-tts.sock")
SR = 24000


def stream_request(sock_path, request, out=None, play=None):
    """Send `request` (dict) to the server and stream the audio.

    play: True -> speakers (sounddevice); None -> speakers unless `out` is set.
    out:  path -> also/instead write a WAV.
    """
    if play is None:
        play = out is None

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall((json.dumps(request) + "\n").encode())
    f = s.makefile("r")

    player = None
    if play:
        try:
            import sounddevice as sd
            player = sd.OutputStream(samplerate=SR, channels=1, dtype="float32")
            player.start()
        except Exception as e:  # no audio device / sounddevice missing
            print(f"[client] audio playback unavailable ({e}); use --out to save a WAV", file=sys.stderr)
            player = None

    chunks = []
    n = 0
    err = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        kind = ev.get("event")
        if kind == "error":
            err = ev.get("message", "unknown error")
            break
        if kind == "audio_chunk":
            if ev.get("done"):
                break
            pcm = np.frombuffer(base64.b64decode(ev["pcm_b64"]), dtype="<f4")
            n += len(pcm)
            if player is not None:
                player.write(pcm)
            if out is not None:
                chunks.append(pcm)

    if player is not None:
        player.stop()
        player.close()
    s.close()

    if err is not None:
        print(f"[client] server error: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"[client] received {n} samples ({n / SR:.2f}s)", file=sys.stderr)
    if out is not None and chunks:
        import soundfile as sf
        sf.write(out, np.concatenate(chunks), SR)
        print(f"[client] wrote {out}", file=sys.stderr)
