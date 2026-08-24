#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump MiniMax Music 3 reference tensors for brain's parity ladder.

MiniMax Music 3 (`MiniMaxAI/MiniMax-Music3`) shipped in **released diffusers
0.40.0**, so this dumper needs nothing but this repo's ordinary
`requirements.txt` set (`diffusers>=0.40`, `torch`, `transformers`,
`safetensors`, `numpy`, plus `huggingface_hub` if you need to fetch a
checkpoint subfolder).

It used to require an unmerged PR branch installed into its own venv, which
is why this file and `requirements.txt` both used to carry a "no official
inference code" note. That is retired: 0.40.0 carries both the four
`MiniMaxMusic3*` model classes and the
`diffusers.modular_pipelines.minimax_music3` modular pipeline, and there is
now an official upstream repo (`MiniMax-AI/MiniMax-Music3`) with a
SGLang-Omni serving path and a reference output WAV. Only the four
`MiniMaxMusic3*`-prefixed classes are needed here (`ConditionEncoder`,
`RVQDepthDecoder`, `Transformer1DModel`, `Vocoder`) - the fifth component,
the Global LLM, is a real `Qwen3ForCausalLM` covered by the standard
`transformers` package and has its own golden path, not this script.

Two independent dim regimes, per component:

  --tiny    random weights at brain's own `::tiny()` config dims (no
            checkpoint needed - the diffusers classes accept arbitrary
            constructor dims, same as this repo's other `*_real_dump_*.py`
            scripts do for a custom config). Exercises the block math.
  --real    real weights, loaded from a downloaded checkpoint directory
            (one of the released repo's `condition_encoder/`, `vocoder/`,
            `rvq_depth_decoder/`, `transformer/` subfolders). Exercises
            import correctness.

Usage:
  python3 tools/goldens/minimaxmusic3_dump_reference.py --component vocoder \
      --tiny --real --real-dir <checkpoint>/vocoder --out testdata/golden/minimaxmusic3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

SEED = 0
COMPONENTS = ("condition_encoder", "vocoder", "depth_decoder", "dit")


def write_f32(path, arr):
    np.asarray(arr, dtype=np.float32).reshape(-1).tofile(path)


def write_u32(path, arr):
    np.asarray(arr, dtype=np.uint32).reshape(-1).tofile(path)


def _weight_files(d):
    return sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.endswith((".safetensors", ".bin", ".pt"))
    )


def save_tiny_state_dict(model, prefix):
    """`--tiny` has no source checkpoint - the model's weights are whatever
    PyTorch's default init drew for this exact `torch.manual_seed(SEED)`
    call, which nothing on the Rust side can reproduce bit-for-bit (a
    different RNG algorithm entirely). So the random weights themselves are
    part of the golden, not just the forward's input/output."""
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, prefix + "_state_dict.safetensors")


# ---- condition encoder -------------------------------------------------

# `::tiny()` in crates/minimaxmusic3/src/config.rs.
TINY_CONDITION = dict(
    condition_hidden_dim=8, num_condition_layers=4, out_dim=6,
    input_sampling_rate=24000, input_hop_length=960,
    output_sampling_rate=44100, output_hop_length=512,
)
REAL_CONDITION = dict(
    condition_hidden_dim=4096, num_condition_layers=8, out_dim=2048,
    input_sampling_rate=24000, input_hop_length=960,
    output_sampling_rate=44100, output_hop_length=512,
)


def dump_condition_encoder(out_dir, cfg, weights_dir, tag):
    from diffusers import MiniMaxMusic3ConditionEncoder

    torch.manual_seed(SEED)
    model = MiniMaxMusic3ConditionEncoder(**cfg)
    prefix = os.path.join(out_dir, f"condition_encoder_{tag}")
    files = []
    if weights_dir:
        sd = load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
        model.load_state_dict(sd, strict=True)
        files = _weight_files(weights_dir)
    else:
        save_tiny_state_dict(model, prefix)
    model = model.float().eval()

    batch, frames = 1, 5
    hidden = torch.randn(batch, frames, cfg["num_condition_layers"] * cfg["condition_hidden_dim"])
    with torch.no_grad():
        out = model(hidden)
    write_f32(prefix + "_in.f32", hidden)
    write_f32(prefix + "_out.f32", out)
    meta = {
        "frames": frames, "batch": batch, **{k: int(v) if isinstance(v, int) else v for k, v in cfg.items()},
        "out_shape": list(out.shape),
        "source": source_block(checkpoint="MiniMaxAI/MiniMax-Music3", files=files, identity={
            k: int(v) for k, v in cfg.items() if isinstance(v, int)
        }),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"condition_encoder[{tag}]: in {tuple(hidden.shape)} -> out {tuple(out.shape)}"
          f" [min,max,mean]=[{out.min():.4f},{out.max():.4f},{out.mean():.4f}]")


# ---- vocoder -------------------------------------------------------------

TINY_VOCODER = dict(latent_channels=4, decoder_input_dim=8, decoder_hidden_dim=16,
                     upsampling_ratios=(2, 2), sampling_rate=8000)
REAL_VOCODER = dict(latent_channels=128, decoder_input_dim=1024, decoder_hidden_dim=1536,
                     upsampling_ratios=(8, 8, 4, 2), sampling_rate=44100)


def dump_vocoder(out_dir, cfg, weights_dir, tag):
    from diffusers import MiniMaxMusic3Vocoder

    torch.manual_seed(SEED)
    model = MiniMaxMusic3Vocoder(**cfg)
    prefix = os.path.join(out_dir, f"vocoder_{tag}")
    files = []
    if weights_dir:
        sd = load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
        model.load_state_dict(sd, strict=True)
        files = _weight_files(weights_dir)
    else:
        save_tiny_state_dict(model, prefix)
    model = model.float().eval()

    batch, length = 1, 6
    latents = torch.randn(batch, cfg["latent_channels"], length)
    with torch.no_grad():
        out = model(latents)
    write_f32(prefix + "_in.f32", latents)
    write_f32(prefix + "_out.f32", out)
    ident = {k: int(v) for k, v in cfg.items() if isinstance(v, int)}
    ident["num_upsample_stages"] = len(cfg["upsampling_ratios"])
    meta = {
        "batch": batch, "length": length, "upsampling_ratios": list(cfg["upsampling_ratios"]),
        "out_shape": list(out.shape),
        "source": source_block(checkpoint="MiniMaxAI/MiniMax-Music3", files=files, identity=ident),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"vocoder[{tag}]: in {tuple(latents.shape)} -> out {tuple(out.shape)}"
          f" [min,max,mean]=[{out.min():.4f},{out.max():.4f},{out.mean():.4f}]")


# ---- RVQ depth decoder ----------------------------------------------------

TINY_DEPTH = dict(hidden_size=8, num_layers=2, num_attention_heads=2, intermediate_size=16,
                   audio_vocab_size=5, num_codebooks=4, max_position_embeddings=9)
REAL_DEPTH = dict(hidden_size=4096, num_layers=4, num_attention_heads=16, intermediate_size=6144,
                   audio_vocab_size=1024, num_codebooks=8, max_position_embeddings=16)


def dump_depth_decoder(out_dir, cfg, weights_dir, tag):
    from diffusers import MiniMaxMusic3RVQDepthDecoder

    torch.manual_seed(SEED)
    model = MiniMaxMusic3RVQDepthDecoder(**cfg)
    prefix = os.path.join(out_dir, f"depth_decoder_{tag}")
    files = []
    if weights_dir:
        sd = load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
        model.load_state_dict(sd, strict=True)
        files = _weight_files(weights_dir)
    else:
        save_tiny_state_dict(model, prefix)
    model = model.float().eval()

    batch, steps = 1, cfg["num_codebooks"]
    inputs_embeds = torch.randn(batch, steps, cfg["hidden_size"])
    codes = torch.randint(0, cfg["audio_vocab_size"], (batch, cfg["num_codebooks"] - 1))
    proj_in = torch.randn(batch, cfg["hidden_size"])
    with torch.no_grad():
        hidden_out = model(inputs_embeds)
        proj_out = model.projection(proj_in)
        offsets = torch.arange(cfg["num_codebooks"] - 1) * cfg["audio_vocab_size"]
        embed_out = model.audio_embeddings(codes + offsets.unsqueeze(0))
        head_ins = torch.randn(cfg["num_codebooks"] - 1, batch, cfg["hidden_size"])
        head_outs = [model.audio_heads[i](head_ins[i]) for i in range(cfg["num_codebooks"] - 1)]

    write_f32(prefix + "_inputs_embeds.f32", inputs_embeds)
    write_f32(prefix + "_hidden_out.f32", hidden_out)
    write_f32(prefix + "_proj_in.f32", proj_in)
    write_f32(prefix + "_proj_out.f32", proj_out)
    write_u32(prefix + "_codes.u32", codes)
    write_f32(prefix + "_embed_out.f32", embed_out)
    write_f32(prefix + "_head_ins.f32", head_ins)
    write_f32(prefix + "_head_outs.f32", torch.stack(head_outs, dim=0))
    ident = {k: int(v) for k, v in cfg.items()}
    meta = {
        "batch": batch, "steps": steps,
        "hidden_out_shape": list(hidden_out.shape),
        "proj_out_shape": list(proj_out.shape),
        "embed_out_shape": list(embed_out.shape),
        "head_outs_shape": list(torch.stack(head_outs, dim=0).shape),
        "source": source_block(checkpoint="MiniMaxAI/MiniMax-Music3", files=files, identity=ident),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"depth_decoder[{tag}]: hidden_out {tuple(hidden_out.shape)}"
          f" [min,max,mean]=[{hidden_out.min():.4f},{hidden_out.max():.4f},{hidden_out.mean():.4f}]")


# ---- flow-matching DiT -----------------------------------------------------

TINY_DIT = dict(in_channels=4, condition_dim=6, num_layers=2, num_attention_heads=2,
                 attention_head_dim=4, ff_inner_dim=16, rotary_dim=2, fourier_embedding_dim=8)
REAL_DIT = dict(in_channels=128, condition_dim=2048, num_layers=36, num_attention_heads=32,
                 attention_head_dim=64, ff_inner_dim=8192, rotary_dim=32, fourier_embedding_dim=256)


def dump_dit(out_dir, cfg, weights_dir, tag):
    from diffusers import MiniMaxMusic3Transformer1DModel

    torch.manual_seed(SEED)
    model = MiniMaxMusic3Transformer1DModel(**cfg)
    prefix = os.path.join(out_dir, f"dit_{tag}")
    files = []
    if weights_dir:
        index_path = os.path.join(weights_dir, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            sd = {}
            for shard in sorted(set(index["weight_map"].values())):
                sd.update(load_file(os.path.join(weights_dir, shard)))
        else:
            sd = load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
        model.load_state_dict(sd, strict=True)
        files = _weight_files(weights_dir)
    else:
        save_tiny_state_dict(model, prefix)
    model = model.float().eval()

    batch, length = 1, 5
    hidden_states = torch.randn(batch, cfg["in_channels"], length)
    timestep = torch.tensor([0.3])
    encoder_hidden_states = torch.randn(batch, length, cfg["condition_dim"])
    with torch.no_grad():
        out = model(hidden_states, timestep, encoder_hidden_states, return_dict=False)[0]

    write_f32(prefix + "_hidden_states.f32", hidden_states)
    write_f32(prefix + "_timestep.f32", timestep)
    write_f32(prefix + "_encoder_hidden_states.f32", encoder_hidden_states)
    write_f32(prefix + "_out.f32", out)
    ident = {k: int(v) for k, v in cfg.items()}
    meta = {
        "batch": batch, "length": length, "out_shape": list(out.shape),
        "source": source_block(checkpoint="MiniMaxAI/MiniMax-Music3", files=files, identity=ident),
    }
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"dit[{tag}]: out {tuple(out.shape)}"
          f" [min,max,mean]=[{out.min():.4f},{out.max():.4f},{out.mean():.4f}]")


DUMPERS = {
    "condition_encoder": (dump_condition_encoder, TINY_CONDITION, REAL_CONDITION),
    "vocoder": (dump_vocoder, TINY_VOCODER, REAL_VOCODER),
    "depth_decoder": (dump_depth_decoder, TINY_DEPTH, REAL_DEPTH),
    "dit": (dump_dit, TINY_DIT, REAL_DIT),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=COMPONENTS, required=True)
    ap.add_argument("--tiny", action="store_true", help="dump the random-weight ::tiny() config")
    ap.add_argument("--real", action="store_true", help="dump real weights from --real-dir")
    ap.add_argument("--real-dir", help="the component's checkpoint subfolder (e.g. .../vocoder)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.tiny and not args.real:
        raise SystemExit("pass --tiny and/or --real")
    if args.real and not args.real_dir:
        raise SystemExit("--real needs --real-dir")

    os.makedirs(args.out, exist_ok=True)
    fn, tiny_cfg, real_cfg = DUMPERS[args.component]
    if args.tiny:
        fn(args.out, tiny_cfg, None, "tiny")
    if args.real:
        fn(args.out, real_cfg, args.real_dir, "real")


if __name__ == "__main__":
    main()
