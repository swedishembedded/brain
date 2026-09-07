#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump real-weight numeric parity goldens for MiniMax-H3's audio VAE decoder.

Loads the checkpoint's OWN shipped reference implementation
(`audio_vae/minimax_h3_audio_vae.py:MiniMaxH3AudioVAE`, Apache-2.0/MIT,
vendored into the checkpoint directory itself - not reimplemented or
reconstructed here) via `from_pretrained`, decodes a short deterministic
synthetic latent, and dumps:

  minimaxh3_audio_vae.safetensors
    z            [C_lat, T]   the input latent fed to decode()
    waveform     [T*hop]      decode(z)[0,0] - the official end-to-end output
    tap_dec_in_proj  [C_mel, T]        dec_in_proj(z) - decode()'s first op
    tap_conv_pre     [C0, T]           decoder.conv_pre(tap_dec_in_proj)
    tap_stage0       [C1, T*rates[0]]  decoder's first upsample+resblock-average
                                       stage output (`x` after loop iteration 0
                                       in BigVGAN.forward), the SAME quantity
                                       `crate::vocoder::decode`'s `h` holds
                                       after its own i=0 loop iteration.
  manifest.json  shapes, sha256, seed, and the `source` block
                 (`tools/goldens/golden_source.py` convention) recording the
                 checkpoint identity that fixes every dumped tensor's shape.

`DacAudioVAE.decode` is exactly `z = dec_in_proj(z); return decoder(z)` (no
encoder call, no mean/std normalization - both out of scope, see
`crate::vocoder`'s own module doc) - the taps above are captured by manually
re-running `decoder`'s stages one at a time on the SAME submodules
`from_pretrained` loaded, rather than by reimplementing any math, and
self-validated against the official `vae.decode(z)` call: this dumper's
handwritten stage replay must reproduce the official end-to-end output
bit-for-bit before anything is written, or it aborts (this is the "compute
the same quantity two ways" self-check porting.md asks for, not merely
trusting the manual replay).

Usage:
  python3 tools/minimaxh3_audio_vae_dump_reference.py \\
      --audio-vae-dir "$BRAIN_MINIMAXH3_DIR/FL2VA/audio_vae" \\
      --out testdata/golden/minimaxh3/audio_vae [--t 6 --seed 7]
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/goldens")
from golden_source import source_block  # noqa: E402


def load_ref_package(component_dir: Path, module_name: str):
    """Import `<component_dir>/<module_name>.py` as though it were installed
    at `<component_dir>` a real Python package - the checkpoint's own files
    use relative imports (`from .dac_activations import SnakeBeta`), which
    only resolve inside a registered package, and this checkpoint directory
    has no `__init__.py` of its own (diffusers' `trust_remote_code` dynamic
    module loader supplies that machinery; this script supplies the same
    trick directly rather than depending on diffusers/transformers for it).
    """
    pkg_name = "_minimaxh3_audio_vae_ref"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(component_dir)]
        sys.modules[pkg_name] = pkg
    full_name = f"{pkg_name}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, component_dir / f"{module_name}.py")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def det_latent(c, t, seed):
    """Deterministic `[1, c, t]` synthetic VAE latent - small values in the
    range real encoded latents occupy (this checkpoint's own `latents_std`
    is ~1.5-3.3 per channel, see `audio_vae/config.json`), NOT drawn from an
    actual encode (encode is out of scope for this port, see
    `crate::vocoder`'s module doc)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((1, c, t), generator=g) * 0.8).to(torch.float32)


def replay_decoder_stages(decoder, x):
    """Re-run `BigVGAN.forward` (`audio_vae/dac_bigvgan.py`) stage by stage on
    the ALREADY-LOADED `decoder` submodules, capturing the output right after
    the first upsample+resblock-average stage. Returns `(stage0, final)` -
    `final` must equal `decoder(x)` bit-for-bit, checked by the caller, since
    this is the same forward computed a second, more granular way rather than
    a different one."""
    h = decoder.conv_pre(x)
    stage0 = None
    for i in range(decoder.num_upsamples):
        for up in decoder.ups[i]:
            h = up(h)
        xs = None
        for j in range(decoder.num_kernels):
            y = decoder.resblocks[i * decoder.num_kernels + j](h)
            xs = y if xs is None else xs + y
        h = xs / decoder.num_kernels
        if i == 0:
            stage0 = h.clone()
    post = decoder.activation_post(h)
    final = decoder.conv_post(post)
    final = torch.tanh(final) if decoder.use_tanh_at_final else torch.clamp(final, -1.0, 1.0)
    return stage0, final


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}" for k, v in sorted(tensors.items())), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-vae-dir", required=True, help="the audio_vae/ component directory (model.safetensors + config.json + the shipped .py sources)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=int, default=6, help="latent frames to decode")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    component_dir = Path(args.audio_vae_dir).resolve()
    safetensors_path = component_dir / "model.safetensors"
    assert safetensors_path.is_file(), f"{safetensors_path} not found"

    mod = load_ref_package(component_dir, "minimax_h3_audio_vae")
    vae = mod.MiniMaxH3AudioVAE.from_pretrained(str(component_dir))
    model = vae.model
    print(f"built DacAudioVAE ({sum(p.numel() for p in model.parameters())} params): "
          f"decoder_dim={model.decoder_dim} vae_latent_channels={model.dec_in_proj.in_channels} "
          f"sample_rate={model.sample_rate}", flush=True)

    z = det_latent(model.dec_in_proj.in_channels, args.t, args.seed)
    print(f"z: {tuple(z.shape)}", flush=True)

    # ---- official end-to-end path -----------------------------------------
    official_wave = vae.decode(z)

    # ---- self-validation: the same forward, replayed stage by stage -------
    proj = model.dec_in_proj(z)
    stage0, replayed_final = replay_decoder_stages(model.decoder, proj)
    d = (replayed_final.double() - official_wave.double()).abs().max().item()
    print(f"  self-validate stage replay vs vae.decode(z): max abs diff {d:.3e}", flush=True)
    assert d == 0.0, f"stage-by-stage replay disagrees with vae.decode(z) by {d:.3e} - not the same forward"

    # ---- self-validation: fresh module instantiation, bit-identical -------
    vae2 = mod.MiniMaxH3AudioVAE.from_pretrained(str(component_dir))
    wave2 = vae2.decode(z)
    d2 = (wave2.double() - official_wave.double()).abs().max().item()
    print(f"  self-validate fresh-instantiation decode: max abs diff {d2:.3e}", flush=True)
    assert d2 == 0.0, f"a freshly re-loaded model disagrees by {d2:.3e} - decode() is not deterministic"
    del vae2, wave2

    tensors = {
        "z": z[0],
        "waveform": official_wave[0, 0],
        "tap_dec_in_proj": proj[0],
        "tap_conv_pre": model.decoder.conv_pre(proj)[0],
        "tap_stage0": stage0[0],
    }

    cfg = model.decoder
    manifest = {
        "run": {"seed": args.seed, "t": args.t, "audio_vae_dir": str(component_dir), "sample_rate": model.sample_rate},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    # identity: the exact fields crate::vocoder::VocoderConfig::h3_32khz()
    # carries - together they fix every tensor shape in this dump (a
    # different decoder_dim/vae_latent_channels tier would produce
    # same-rank, wrong-width tensors, the mismatch this block exists to catch
    # rather than mis-report as a parity failure). hash_files=True: the real
    # checkpoint is 578MB, well within a short dump's hashing budget.
    manifest["source"] = source_block(
        checkpoint="MiniMaxAI/MiniMax-H3",
        files=[str(safetensors_path)],
        identity={
            "vae_latent_channels": int(model.dec_in_proj.in_channels),
            "mel_channels": int(model.dec_in_proj.out_channels),
            "upsample_initial_channel": int(cfg.conv_pre.out_channels),
            "num_upsamples": int(cfg.num_upsamples),
            "out_channels": int(cfg.conv_post.out_channels),
            "sample_rate": int(model.sample_rate),
        },
    )
    save(args.out, "minimaxh3_audio_vae.safetensors", tensors, manifest)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
