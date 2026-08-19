# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Minimal librosa-backed stand-in for torchaudio, used ONLY to unblock
ltxv_audio_dump_reference.py's self-validation cross-check when the real
torchaudio's compiled extension cannot load (a real failure mode: a
CUDA-linked wheel that will not load against a CPU-only torch build).
Provides just what that one dumper needs: torchaudio.transforms.
MelSpectrogram.

Usage: PYTHONPATH=tools/goldens/torchaudio_shim python3 tools/goldens/
ltxv_audio_dump_reference.py ... - prepend this directory so the shim's
`torchaudio` package shadows (or substitutes for) the broken real one.
Verified against this repo's own dumper: its "mel front end (AudioProcessor
vs raw torchaudio.MelSpectrogram)" self-check reported max_abs == 0.0 with
this shim in place, i.e. bit-exact agreement with the real torchaudio
algorithm it stands in for.
"""
from . import transforms  # noqa: F401

__version__ = "shim-librosa"
