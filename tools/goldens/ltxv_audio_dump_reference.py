#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 audio VAE + BigVGAN vocoder reference goldens (NO BWE stage).

Runs the OFFICIAL `ltx_core.model.audio_vae.{audio_vae,vocoder}` (CPU, fp32)
with the REAL `ltx-2.5-audio-vae-bf16.safetensors` weights on a deterministic
synthetic 16kHz stereo waveform, dumping every boundary:

  audio.safetensors   waveform -> mel front end -> encoder stage taps -> latent
                      -> decoder stage taps -> reconstructed mel -> vocoder
                      stage taps -> reconstructed waveform
  manifest.json        shapes, sha256, run params, library versions

The bandwidth-extension (BWE) stage and its checkpoint-basis causal STFT
(`MelSTFT`/`VocoderWithBWE`) are explicitly OUT OF SCOPE for this M1 milestone
- only the base `Vocoder` (checkpoint config `vocoder.vocoder`) is dumped,
never `vocoder.bwe_generator`/`vocoder.mel_stft`.

## Weight loading

`ltx_core.model.audio_vae.model_configurator` ships `AudioEncoderConfigurator`
/ `AudioDecoderConfigurator` (build from the checkpoint's `config.audio_vae`
JSON) and `AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER` /
`AUDIO_VAE_DECODER_COMFY_KEYS_FILTER` (`SDOps` renaming that strips the
checkpoint's `audio_vae.encoder.` / `audio_vae.decoder.` prefixes) - loading
through these is `state_dict(strict=True)` with ZERO missing/unexpected keys,
verified empirically. The library's own `VOCODER_COMFY_KEYS_FILTER` targets a
`VocoderWithBWE` wrapper (`vocoder.*` -> `vocoder.*`/`bwe_generator.*`/
`mel_stft.*`), which is the wrong shape for a bare `Vocoder`; this dumper adds
one small local `SDOps` (`_VOCODER_BASE_KEYS_FILTER`) that strips the doubled
`vocoder.vocoder.` prefix down to the bare names `Vocoder.state_dict()`
expects (also verified empirically: zero missing/unexpected keys, ignoring
`vocoder.bwe_generator.*`/`vocoder.mel_stft.*` by construction since those
never match the `vocoder.vocoder.` prefix).

## Self-validation inside the dumper

1. **Mel front end, two independent code paths**: `AudioProcessor.waveform_to_mel`
   (what the encoder actually calls) vs a raw `torchaudio.transforms.MelSpectrogram`
   built directly from the milestone's stated config
   (`sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160, f_min=0.0,
   f_max=8000, n_mels=64, power=1.0, mel_scale="slaney", norm="slaney"`) plus the
   same `log(clamp(mel, min=1e-5))` compression, applied to the SAME waveform.
   Asserted to agree exactly (both are the same op, called two different ways).
2. **Fresh-module determinism**: encoder/decoder/vocoder are built and loaded a
   SECOND time from scratch and the whole run repeated; asserted bit-identical
   (no dropout/randomness is active at `eval()`).

Usage:
  python tools/goldens/ltxv_audio_dump_reference.py \\
      --weights /path/to/ltx-2.5-audio-vae-bf16.safetensors \\
      --out testdata/golden/ltxv/audio [--duration 1.0 --seed 7]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import save_file

# `LTXV_REFERENCE_ROOT` overrides for a checkout elsewhere; the default is
# repo-relative (`scratchpad/reference/ltxv/`, gitignored), never a
# machine-specific absolute path.
_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

from ltx_core.loader.sd_ops import SDOps  # noqa: E402
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402
from ltx_core.model.audio_vae.model_configurator import (  # noqa: E402
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    _vocoder_from_config,
)
from ltx_core.model.audio_vae.ops import AudioProcessor  # noqa: E402
from ltx_core.types import Audio  # noqa: E402

# Strips the doubled `vocoder.vocoder.` checkpoint prefix down to the bare
# names `Vocoder.state_dict()` expects (`conv_pre.*`, `ups.*`, `resblocks.*`,
# `act_post.*`, `conv_post.*`). Deliberately narrower than the library's
# `VOCODER_COMFY_KEYS_FILTER` (which targets `VocoderWithBWE`, out of scope
# here) - `vocoder.bwe_generator.*` / `vocoder.mel_stft.*` never match this
# prefix, so BWE weights are never touched.
_VOCODER_BASE_KEYS_FILTER = (
    SDOps("VOCODER_BASE_KEYS_FILTER")
    .with_matching(prefix="vocoder.vocoder.")
    .with_replacement("vocoder.vocoder.", "")
)


def det_waveform(channels, samples, seed):
    """Deterministic stereo waveform in [-1, 1]: a few harmonics + small noise,
    envelope-shaped so it is not a pure discontinuous tone (avoids a degenerate
    all-silence or clipped-square-wave mel spectrogram)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, 1.0, samples)
    sig = (0.5 * torch.sin(2 * torch.pi * 220.0 * t)
           + 0.25 * torch.sin(2 * torch.pi * 660.0 * t)
           + 0.1 * torch.sin(2 * torch.pi * 1500.0 * t))
    envelope = torch.clamp(4.0 * t * (1.0 - t), 0.0, 1.0) ** 0.5
    sig = sig * envelope
    left = sig
    right = sig * 0.8 + 0.2 * torch.roll(sig, shifts=samples // 37)
    wav = torch.stack([left, right][:channels], dim=0)
    wav = wav + 0.01 * torch.randn(wav.shape, generator=g)
    return wav.clamp(-1.0, 1.0).unsqueeze(0).contiguous()


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                      "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}"
                                        for k, v in sorted(tensors.items())), flush=True)


def agree(name, a, b, tol=1e-6):
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    print(f"  self-validate {name}: max abs {d:.3e} / scale {scale:.3g} "
          f"= {rel:.2e} (tol {tol:g})", flush=True)
    assert rel <= tol, f"{name}: disagree by {rel:.3e} relative"
    return d


class Taps:
    def __init__(self):
        self.acc, self.handles = {}, []

    def watch(self, name, module):
        def hook(_m, _i, o):
            self.acc[name] = o.detach().clone()
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_models(weights_path):
    """Build+load a FRESH encoder/decoder/vocoder triple from the real checkpoint."""
    loader = SafetensorsModelStateDictLoader()
    metadata = loader.metadata(weights_path)

    encoder = AudioEncoderConfigurator.from_metadata(metadata)
    decoder = AudioDecoderConfigurator.from_metadata(metadata)

    vocoder_meta = metadata.get("config", {}).get("vocoder", {})
    vocoder_cfg = vocoder_meta.get("vocoder", {})
    bwe_cfg = vocoder_meta.get("bwe")
    assert bwe_cfg is not None, (
        "checkpoint has no config.vocoder.bwe - expected the ltx-2.5 nested "
        "vocoder+bwe layout (output_sampling_rate for the base vocoder is "
        "read from bwe.input_sampling_rate)"
    )
    vocoder = _vocoder_from_config(vocoder_cfg, output_sampling_rate=bwe_cfg["input_sampling_rate"])

    sd_enc = loader.load(weights_path, AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER)
    sd_dec = loader.load(weights_path, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER)
    sd_voc = loader.load(weights_path, _VOCODER_BASE_KEYS_FILTER)
    encoder.load_state_dict({k: v.to(torch.float32) for k, v in sd_enc.sd.items()}, strict=True)
    decoder.load_state_dict({k: v.to(torch.float32) for k, v in sd_dec.sd.items()}, strict=True)
    vocoder.load_state_dict({k: v.to(torch.float32) for k, v in sd_voc.sd.items()}, strict=True)
    encoder.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    vocoder.eval().requires_grad_(False)

    ddconfig = metadata["config"]["audio_vae"]["model"]["params"]["ddconfig"]
    return encoder, decoder, vocoder, ddconfig


def watch_audio_encoder(encoder, taps):
    taps.watch("enc.conv_in", encoder.conv_in)
    for level in range(encoder.num_resolutions):
        stage = encoder.down[level]
        for i, block in enumerate(stage.block):
            taps.watch(f"enc.down.{level}.block.{i}", block)
        if hasattr(stage, "downsample"):
            taps.watch(f"enc.down.{level}.downsample", stage.downsample)
    taps.watch("enc.mid.block_1", encoder.mid.block_1)
    taps.watch("enc.mid.block_2", encoder.mid.block_2)
    taps.watch("enc.norm_out", encoder.norm_out)
    taps.watch("enc.conv_out", encoder.conv_out)


def watch_audio_decoder(decoder, taps):
    taps.watch("dec.conv_in", decoder.conv_in)
    taps.watch("dec.mid.block_1", decoder.mid.block_1)
    taps.watch("dec.mid.block_2", decoder.mid.block_2)
    for level in reversed(range(decoder.num_resolutions)):
        stage = decoder.up[level]
        for i, block in enumerate(stage.block):
            taps.watch(f"dec.up.{level}.block.{i}", block)
        if hasattr(stage, "upsample"):
            taps.watch(f"dec.up.{level}.upsample", stage.upsample)
    taps.watch("dec.norm_out", decoder.norm_out)
    taps.watch("dec.conv_out", decoder.conv_out)


def watch_vocoder(vocoder, taps):
    taps.watch("voc.conv_pre", vocoder.conv_pre)
    for i, up in enumerate(vocoder.ups):
        taps.watch(f"voc.ups.{i}", up)
    for i, rb in enumerate(vocoder.resblocks):
        taps.watch(f"voc.resblocks.{i}", rb)
    taps.watch("voc.act_post", vocoder.act_post)
    taps.watch("voc.conv_post", vocoder.conv_post)


def run_round_trip(encoder, decoder, vocoder, audio_processor, waveform):
    audio = Audio(waveform=waveform, sampling_rate=audio_processor.target_sample_rate)
    mel = audio_processor.waveform_to_mel(audio)

    etaps = Taps()
    watch_audio_encoder(encoder, etaps)
    latent = encoder(mel)
    etaps.close()

    dtaps = Taps()
    watch_audio_decoder(decoder, dtaps)
    recon_mel = decoder(latent)
    dtaps.close()

    vtaps = Taps()
    watch_vocoder(vocoder, vtaps)
    recon_wave = vocoder(recon_mel)
    vtaps.close()

    return {"mel": mel, "latent": latent, "recon_mel": recon_mel, "recon_wave": recon_wave,
            "enc_taps": dict(etaps.acc), "dec_taps": dict(dtaps.acc), "voc_taps": dict(vtaps.acc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="ltx-2.5-audio-vae-bf16.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=1.0, help="seconds of synthetic audio")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--taps", action="store_true", help="also dump every per-block activation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    encoder, decoder, vocoder, ddconfig = build_models(args.weights)
    print(f"built encoder ({sum(p.numel() for p in encoder.parameters())} params), "
          f"decoder ({sum(p.numel() for p in decoder.parameters())} params), "
          f"vocoder ({sum(p.numel() for p in vocoder.parameters())} params)", flush=True)

    audio_processor = AudioProcessor(target_sample_rate=encoder.sample_rate, mel_bins=encoder.mel_bins,
                                     mel_hop_length=encoder.mel_hop_length, n_fft=encoder.n_fft)

    samples = int(round(args.duration * encoder.sample_rate))
    waveform = det_waveform(2, samples, args.seed)
    print(f"waveform: {tuple(waveform.shape)} @ {encoder.sample_rate} Hz "
          f"({args.duration}s)", flush=True)

    r1 = run_round_trip(encoder, decoder, vocoder, audio_processor, waveform)

    # ---- self-validation 1: mel front end, two independent code paths ------
    ref_mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160,
        f_min=0.0, f_max=8000, n_mels=64, power=1.0, mel_scale="slaney", norm="slaney")
    ref_mel = ref_mel_transform(waveform)
    ref_mel = torch.log(torch.clamp(ref_mel, min=1e-5)).permute(0, 1, 3, 2).contiguous()
    agree("mel front end (AudioProcessor vs raw torchaudio.MelSpectrogram)", r1["mel"], ref_mel)

    # ---- self-validation 2: fresh module instantiation, bit-identical ------
    encoder2, decoder2, vocoder2, _ = build_models(args.weights)
    audio_processor2 = AudioProcessor(target_sample_rate=encoder2.sample_rate, mel_bins=encoder2.mel_bins,
                                      mel_hop_length=encoder2.mel_hop_length, n_fft=encoder2.n_fft)
    r2 = run_round_trip(encoder2, decoder2, vocoder2, audio_processor2, waveform)
    agree("fresh-instantiation latent", r2["latent"], r1["latent"], tol=0.0)
    agree("fresh-instantiation recon_wave", r2["recon_wave"], r1["recon_wave"], tol=0.0)
    del encoder2, decoder2, vocoder2

    cos_mel = F.cosine_similarity(r1["recon_mel"].flatten().double(),
                                  r1["mel"].flatten().double(), dim=0).item()
    print(f"  round-trip cosine(recon_mel, mel): {cos_mel:.6f}", flush=True)

    tensors = {
        "waveform": waveform[0],
        "mel": r1["mel"][0],
        "latent": r1["latent"][0],
        "recon_mel": r1["recon_mel"][0],
        "recon_wave": r1["recon_wave"][0],
    }
    if args.taps:
        for k, v in r1["enc_taps"].items():
            tensors[f"tap_{k}"] = v[0]
        for k, v in r1["dec_taps"].items():
            tensors[f"tap_{k}"] = v[0]
        for k, v in r1["voc_taps"].items():
            tensors[f"tap_{k}"] = v[0]

    manifest = {
        "run": {"seed": args.seed, "duration": args.duration, "samples": samples,
                "weights": os.path.abspath(args.weights),
                "sample_rate": encoder.sample_rate, "mel_bins": encoder.mel_bins,
                "mel_hop_length": encoder.mel_hop_length, "n_fft": encoder.n_fft,
                "ddconfig": ddconfig,
                "vocoder_output_sampling_rate": vocoder.output_sampling_rate,
                "round_trip_cosine_mel": cos_mel,
                "bwe_in_scope": False},
        "versions": {"torch": torch.__version__, "torchaudio": torchaudio.__version__,
                     "python": sys.version.split()[0]},
    }
    save(args.out, "audio.safetensors", tensors, manifest)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
