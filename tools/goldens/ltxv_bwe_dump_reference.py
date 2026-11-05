#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 vocoder + bandwidth-extension (BWE) reference goldens.

Runs the OFFICIAL `ltx_core.model.audio_vae.vocoder.VocoderWithBWE` (CPU,
fp32), built by the library's own `VocoderConfigurator.from_metadata` and
loaded through its own `VOCODER_COMFY_KEYS_FILTER` with the REAL
`ltx-2.5-audio-vae-bf16.safetensors` weights, on the log-mel of a
deterministic synthetic 16 kHz stereo waveform (the audio VAE's own
`AudioProcessor` front end, so the input is in the domain the vocoder sees
in a real decode). Every boundary of `VocoderWithBWE.forward` is dumped:

  bwe.safetensors   mel_in [2,T,64] -> base_wave [2,L] (16 kHz) ->
                    bwe_mel [2,64,F] (causal checkpoint-basis STFT log-mel
                    of the hop-padded base wave) -> residual [2,3*Lp]
                    (bwe_generator, no final activation) -> skip [2,3*Lp]
                    (Hann-sinc x3 resampler) -> out [2,3*L] (48 kHz)
  chain.safetensors latent [8,Ta,16] (the audio VAE encoder's own latent of
                    the same waveform) -> recon_wave_48k [2,N]: the whole
                    production decode, `decode_audio(latent, AudioDecoder,
                    VocoderWithBWE)`, the reference's own entry point
  manifest.json      shapes, sha256, run params, the source block

Self-validation inside the dumper:

1. The composed boundaries reproduce `VocoderWithBWE.forward` exactly
   (`clamp(residual + skip, -1, 1)[..., :3L]` == the module's own output).
2. A fresh build + load of the module repeats the run bit-identically.

Usage:
  python tools/goldens/ltxv_bwe_dump_reference.py \\
      --weights /path/to/ltx-2.5-audio-vae-bf16.safetensors \\
      --out testdata/golden/ltxv/bwe --seed 7 [--duration 0.5]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

# `LTXV_REFERENCE_ROOT` overrides for a checkout elsewhere; the default is
# repo-relative (`scratchpad/reference/ltxv/`, gitignored).
_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

from ltx_core.model.audio_vae.audio_vae import decode_audio  # noqa: E402
from ltx_core.model.audio_vae.model_configurator import (  # noqa: E402
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae.ops import AudioProcessor  # noqa: E402
from ltx_core.model.audio_vae.vocoder import VocoderWithBWE  # noqa: E402
from ltx_core.types import Audio  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402


def det_waveform(channels, samples, seed):
    """A deterministic two-tone-plus-noise stereo test signal in [-1, 1]."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(samples, dtype=torch.float32) / 16000.0
    sig = 0.4 * torch.sin(2 * torch.pi * 440.0 * t) + 0.2 * torch.sin(2 * torch.pi * 1870.0 * t)
    left = sig
    right = sig * 0.8 + 0.2 * torch.roll(sig, shifts=samples // 37)
    wav = torch.stack([left, right][:channels], dim=0)
    wav = wav + 0.01 * torch.randn(wav.shape, generator=g)
    return wav.clamp(-1.0, 1.0).unsqueeze(0).contiguous()


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in sorted(tensors.items())), flush=True)


def build(weights_path):
    """Build `VocoderWithBWE` from the checkpoint's own config and load it
    strictly. Keys are mapped the way the library's `VOCODER_COMFY_KEYS_FILTER`
    does: select `vocoder.*` and strip that prefix exactly once."""
    with safe_open(weights_path, framework="pt") as f:
        metadata = {"config": json.loads((f.metadata() or {})["config"])}
        sd = {k.removeprefix("vocoder."): f.get_tensor(k).to(torch.float32) for k in f.keys() if k.startswith("vocoder.")}
    model = VocoderConfigurator.from_metadata(metadata)
    assert isinstance(model, VocoderWithBWE), "expected the ltx-2.5 nested vocoder+bwe checkpoint"
    model.load_state_dict(sd, strict=True)
    model.eval().requires_grad_(False)
    return model, metadata


def load_prefixed(weights_path, prefixes):
    """The checkpoint tensors under `prefixes` (old -> new), fp32 - the
    library's `AUDIO_VAE_{ENCODER,DECODER}_COMFY_KEYS_FILTER` mapping."""
    out = {}
    with safe_open(weights_path, framework="pt") as f:
        for k in f.keys():
            for old, new in prefixes:
                if k.startswith(old):
                    out[new + k[len(old):]] = f.get_tensor(k).to(torch.float32)
    return out


def build_codec(weights_path, metadata):
    stats = ("audio_vae.per_channel_statistics.", "per_channel_statistics.")
    encoder = AudioEncoderConfigurator.from_metadata(metadata)
    encoder.load_state_dict(load_prefixed(weights_path, [("audio_vae.encoder.", ""), stats]), strict=True)
    decoder = AudioDecoderConfigurator.from_metadata(metadata)
    decoder.load_state_dict(load_prefixed(weights_path, [("audio_vae.decoder.", ""), stats]), strict=True)
    return encoder.eval().requires_grad_(False), decoder.eval().requires_grad_(False)


def run(model, mel_in):
    """`VocoderWithBWE.forward`, boundary by boundary, plus the module call."""
    base = model.vocoder(mel_in)
    length = base.shape[-1]
    rem = length % model.hop_length
    padded = F.pad(base, (0, model.hop_length - rem)) if rem else base
    bwe_mel = model._compute_mel(padded)
    residual = model.bwe_generator(bwe_mel.transpose(2, 3))
    skip = model.resampler(padded)
    out_len = length * model.output_sampling_rate // model.input_sampling_rate
    composed = torch.clamp(residual + skip, -1, 1)[..., :out_len]
    whole = model(mel_in)
    return {"base_wave": base, "bwe_mel": bwe_mel, "residual": residual, "skip": skip, "out": composed, "whole": whole}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="ltx-2.5-audio-vae-bf16.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=0.5, help="seconds of synthetic audio")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    model, metadata = build(args.weights)
    encoder, decoder = build_codec(args.weights, metadata)
    processor = AudioProcessor(target_sample_rate=encoder.sample_rate, mel_bins=encoder.mel_bins,
                               mel_hop_length=encoder.mel_hop_length, n_fft=encoder.n_fft)
    samples = int(round(args.duration * encoder.sample_rate))
    waveform = det_waveform(2, samples, args.seed)
    mel_in = processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=encoder.sample_rate))  # [1, 2, T, 64]
    print(f"mel_in {tuple(mel_in.shape)}", flush=True)

    r1 = run(model, mel_in)
    d = (r1["out"] - r1["whole"]).abs().max().item()
    print(f"  self-validate composed vs VocoderWithBWE.forward: max abs {d:.3e}", flush=True)
    assert d == 0.0, "the dumped boundaries must compose to the module's own output"

    model2, _ = build(args.weights)
    r2 = run(model2, mel_in)
    assert torch.equal(r2["out"], r1["out"]), "a fresh build must repeat the run bit-identically"

    tensors = {"mel_in": mel_in[0]}
    for k in ("base_wave", "bwe_mel", "residual", "skip", "out"):
        tensors[k] = r1[k][0]
    manifest = {}
    save(args.out, "bwe.safetensors", tensors, manifest)

    latent = encoder(mel_in)
    decoded = decode_audio(latent, decoder, model)
    assert decoded.sampling_rate == 48000, decoded.sampling_rate
    save(args.out, "chain.safetensors", {"latent": latent[0], "recon_wave_48k": decoded.waveform}, manifest)
    bwe_cfg = metadata["config"]["vocoder"]["bwe"]
    manifest["run"] = {"seed": args.seed, "duration": args.duration, "samples": samples,
                       "input_sampling_rate": bwe_cfg["input_sampling_rate"],
                       "output_sampling_rate": bwe_cfg["output_sampling_rate"],
                       "hop_length": bwe_cfg["hop_length"], "n_fft": bwe_cfg["n_fft"],
                       "num_mels": bwe_cfg["num_mels"], "torch": torch.__version__}
    manifest["source"] = source_block(
        checkpoint="Lightricks/LTX-2.5",
        files=[args.weights],
        identity={"bwe_upsample_initial_channel": bwe_cfg["upsample_initial_channel"],
                  "bwe_num_upsamples": len(bwe_cfg["upsample_rates"]),
                  "num_mels": bwe_cfg["num_mels"], "n_fft": bwe_cfg["n_fft"]})
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("wrote manifest.json", flush=True)


if __name__ == "__main__":
    main()
