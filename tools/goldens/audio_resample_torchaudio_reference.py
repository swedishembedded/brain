#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Print the reference resamples `crates/audio/tests/it/resample_torchaudio.rs`
embeds: `torchaudio.transforms.Resample(orig, new)` with its defaults
(`sinc_interp_hann`, `lowpass_filter_width=6`, `rolloff=0.99`, kernel built in
float64 and cached as float32) over a fixed 97-sample fixture, for a
downsample with a large reduced ratio (44100 -> 32000), a plain 2x upsample
and a 3:2 downsample.

    python3 tools/goldens/audio_resample_torchaudio_reference.py

and paste the printed Rust constants into the test.
"""
import math

import torch
import torchaudio


def fixture() -> torch.Tensor:
    """`0.6 sin(0.21 i) + 0.3 cos(1.7 i + 0.4)`, 97 samples, float32."""
    i = torch.arange(97, dtype=torch.float64)
    return (0.6 * torch.sin(0.21 * i) + 0.3 * torch.cos(1.7 * i + 0.4)).to(torch.float32)


def main() -> None:
    x = fixture()[None]
    for orig, new in [(44100, 32000), (16000, 32000), (48000, 32000)]:
        y = torchaudio.transforms.Resample(orig, new)(x)[0]
        body = ", ".join(f"{v:.9e}" for v in y.tolist())
        print(f"const TORCHAUDIO_{orig}_TO_{new}: [f32; {y.numel()}] = [{body}];")


if __name__ == "__main__":
    main()
