#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Qwen3-TTS 12 Hz codec goldens for `crates/mimi`'s parity gates.

The reference is the upstream `qwen-tts` package's `Qwen3TTSTokenizerV2Model`
(see `qwen3tts_ref.py` for why it cannot be `transformers`), run in fp32 on the
CPU against the released `Qwen3-TTS-Tokenizer-12Hz` checkpoint. Two dumps:

  `codec_ref/`      decode side: `codes.bin` [T,16] u32 -> `waveform.bin` f32,
                    read by `crates/mimi/tests/decode.rs`. The codes are the
                    reference ENCODER's own output on a real speech clip, not a
                    random draw, so the decode gate runs on a code sequence the
                    model actually produces.
  `codec_enc_ref/`  encode side: `wav.bin` f32 -> `codes.bin` [T,16] u32, read
                    by `crates/mimi/tests/encode.rs`.

Both `.bin` files are `<u64 LE element count><payload LE>`, the format those
two suites already parse.

`--frames` stays at or below the reference decoder's 300-frame chunk size, so
`chunked_decode` degenerates to a single `forward` and the golden is a
single-shot decode - which is what `mimi::Codec::decode` implements.

`--wav` is any 24 kHz mono clip. The dumps in this repo were made from the
voice-clone example the Qwen3-TTS model card itself links,
`Qwen3-TTS-Repo/clone.wav` on `qianwen-res.oss-cn-beijing.aliyuncs.com`
(sha256 480f55f4...5a6b5c, 24 kHz mono float, first 2 s used) - real speech, and
upstream's own, so re-dumping does not depend on a clip that exists on one
machine. The encode golden carries those samples verbatim in `wav.bin`, so the
encode gate needs no external audio at all.

Usage:
  python3 tools/goldens/qwen3tts_codec_dump_reference.py \
      --ckpt testdata/tts/ckpt/Qwen3-TTS-Tokenizer-12Hz \
      --wav <a 24 kHz mono clip> --out testdata/tts/dumps
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402
from qwen3tts_ref import load_codec  # noqa: E402

CHECKPOINT = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
# `Qwen3TTSTokenizerV2Decoder.chunked_decode`'s default chunk.
MAX_SINGLE_SHOT_FRAMES = 300


def write_prefixed(path, arr, dtype):
    """`<u64 LE count><payload>` - the layout `crates/mimi`'s suites read."""
    a = np.asarray(arr, dtype=dtype).reshape(-1)
    with open(path, "wb") as f:
        f.write(np.uint64(a.size).tobytes())
        f.write(a.tobytes())
    return int(a.size)


def identity_of(cfg):
    """The config fields that fix every dumped tensor's shape. `num_quantizers`
    and `codebook_size` fix the code array, `decode_upsample_rate` fixes the
    waveform length, and the two hidden sizes separate this tier of the codec
    from any other."""
    dec, enc = cfg.decoder_config, cfg.encoder_config
    return {
        "encoder_valid_num_quantizers": cfg.encoder_valid_num_quantizers,
        "decode_upsample_rate": cfg.decode_upsample_rate,
        "encode_downsample_rate": cfg.encode_downsample_rate,
        "num_quantizers": dec.num_quantizers,
        "codebook_size": dec.codebook_size,
        "decoder_hidden_size": dec.hidden_size,
        "decoder_dim": dec.decoder_dim,
        "latent_dim": dec.latent_dim,
        "encoder_hidden_size": enc.hidden_size,
        "encoder_num_hidden_layers": enc.num_hidden_layers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Qwen3-TTS-Tokenizer-12Hz directory")
    ap.add_argument("--wav", required=True, help="24 kHz mono reference clip")
    ap.add_argument("--out", required=True, help="dump root, e.g. testdata/tts/dumps")
    ap.add_argument("--seconds", type=float, default=2.0, help="clip length to encode")
    ap.add_argument("--frames", type=int, default=24, help="frames in the decode golden")
    args = ap.parse_args()

    if args.frames > MAX_SINGLE_SHOT_FRAMES:
        raise SystemExit(
            f"--frames {args.frames} exceeds the reference decoder's {MAX_SINGLE_SHOT_FRAMES}-frame "
            "chunk, so the golden would be a chunked decode with left context and would no longer "
            "describe a single-shot decode"
        )

    config_class, model_class = load_codec()
    import soundfile as sf
    import torch

    cfg = config_class.from_pretrained(args.ckpt)
    model = model_class.from_pretrained(args.ckpt, config=cfg, dtype=torch.float32).eval()

    wav, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != cfg.input_sample_rate:
        raise SystemExit(f"{args.wav} is {sr} Hz; the codec needs {cfg.input_sample_rate} Hz")
    wav = np.ascontiguousarray(wav[: int(args.seconds * sr)], dtype=np.float32)

    x = torch.from_numpy(wav)[None, :]
    # An integer mask: the reference derives the valid frame count from
    # `mask.sum()` and uses it as a slice index.
    mask = torch.ones_like(x, dtype=torch.long)
    with torch.no_grad():
        codes = model.encode(x, padding_mask=mask).audio_codes[0]  # [T, Q]
    n_q = int(codes.shape[1])

    frames = min(args.frames, int(codes.shape[0]))
    decode_codes = codes[:frames]
    with torch.no_grad():
        # [B, Q, T] is what the decoder takes; at <= 300 frames `chunked_decode`
        # is exactly one `forward`, so call the module directly.
        waveform = model.decoder(decode_codes.transpose(0, 1)[None, ...].contiguous()).squeeze(1)[0]

    weights = [os.path.join(args.ckpt, "model.safetensors")]
    source = source_block(checkpoint=CHECKPOINT, files=weights, identity=identity_of(cfg))

    dec_dir = os.path.join(args.out, "codec_ref")
    os.makedirs(dec_dir, exist_ok=True)
    n_codes = write_prefixed(
        os.path.join(dec_dir, "codes.bin"), decode_codes.numpy(), np.uint32
    )
    n_wav = write_prefixed(
        os.path.join(dec_dir, "waveform.bin"), waveform.numpy(), np.float32
    )
    with open(os.path.join(dec_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "frames": frames,
                "num_quantizers": n_q,
                "codes": [frames, n_q],
                "waveform": [n_wav],
                "sample_rate": int(cfg.output_sample_rate),
                "chunked": False,
                "source": source,
            },
            f,
            indent=2,
        )

    enc_dir = os.path.join(args.out, "codec_enc_ref")
    os.makedirs(enc_dir, exist_ok=True)
    n_in = write_prefixed(os.path.join(enc_dir, "wav.bin"), wav, np.float32)
    write_prefixed(os.path.join(enc_dir, "codes.bin"), codes.numpy(), np.uint32)
    with open(os.path.join(enc_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "samples": n_in,
                "frames": int(codes.shape[0]),
                "num_quantizers": n_q,
                "sample_rate": int(cfg.input_sample_rate),
                "source": source,
            },
            f,
            indent=2,
        )

    print(f"decode golden: {frames} frames x {n_q} codes -> {n_wav} samples ({dec_dir})")
    print(f"encode golden: {n_in} samples -> {int(codes.shape[0])} frames x {n_q} ({enc_dir})")
    print(f"waveform rms {float((waveform ** 2).mean().sqrt()):.4f}, peak {float(waveform.abs().max()):.4f}")
    print(f"n_codes written: {n_codes}")


if __name__ == "__main__":
    main()
