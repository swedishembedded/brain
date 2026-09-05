#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump REAL-WEIGHT numeric parity goldens for MiniMax-H3's DiT core.

Unlike `minimaxh3_dit_dump_reference.py` (tiny config, small seeded RANDOM
weights - it needs no checkpoint at all, and by its own design cannot catch
a bug that only manifests with real, structured weights: a wrong transpose,
a wrong axis order, a wrong tensor-name-to-role mapping all look "fine" on
random data since every element is equally arbitrary), this dumper loads
the REAL, installed `diffusers==0.40.0` `MiniMaxH3Transformer3DModel` class
AT THE REAL CHECKPOINT'S OWN CONFIG AND WEIGHTS (`--checkpoint <dir>/transformer`),
and runs ONE forward over a SMALL synthetic packed sequence (the sequence
LENGTH is kept small deliberately - this dumper exists to check that each
LAYER's math is right at real width/depth, not to time a real-resolution
run) spanning all three modalities at two distinct timesteps, exactly
mirroring `minimaxh3_dit_dump_reference.py`'s own input-construction shape
but at the real checkpoint's real per-token widths (text_dim=5120,
video_patch_dim=96, audio_in_channels=32) and real depth (50 layers, not 2).

Dumps ONLY the forward inputs and intermediate taps - NOT the weights (the
real checkpoint is 66GB; both this dumper and the Rust side that reads this
golden already have their own path to the SAME real checkpoint directory,
so there is nothing to gain and 66GB to lose by re-serializing them here).
Taps: block 0 (input/attn-out/out - matches the tiny dumper exactly), PLUS
a middle block (index num_layers//2) and the LAST block, so a mismatch that
only shows up after several blocks' worth of accumulated state (e.g. a
per-block AdaLN indexing bug) localizes to a specific block rather than
only showing up in the final output.

Usage:
  python3 tools/minimaxh3_dit_real_dump_reference.py \\
      --checkpoint "$BRAIN_MINIMAXH3_DIR/transformer" \\
      --out testdata/golden/minimaxh3/dit_real [--seed 3]
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/goldens")
from golden_source import source_block  # noqa: E402

from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel  # noqa: E402

NUM_TEXT = 3
NUM_AUDIO = 2
NUM_VIDEO = 4
SEQ_LEN = NUM_TEXT + NUM_AUDIO + NUM_VIDEO
TEXT_INDICES = list(range(0, NUM_TEXT))
AUDIO_INDICES = list(range(NUM_TEXT, NUM_TEXT + NUM_AUDIO))
VIDEO_INDICES = list(range(NUM_TEXT + NUM_AUDIO, SEQ_LEN))
TIMESTEPS = [0.2, 0.8]


def build_inputs(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    video_patch_dim = cfg.in_channels * cfg.patch_size[0] * cfg.patch_size[1] * cfg.patch_size[2]

    hidden_states = torch.randn((1, NUM_VIDEO, video_patch_dim), generator=g) * 0.3
    audio_hidden_states = torch.randn((1, NUM_AUDIO, cfg.audio_in_channels), generator=g) * 0.3
    encoder_hidden_states = torch.randn((1, NUM_TEXT, cfg.text_dim), generator=g) * 0.3

    token_tags = torch.zeros(SEQ_LEN, dtype=torch.long)
    timestep_indices = torch.zeros(SEQ_LEN, dtype=torch.long)
    for i in TEXT_INDICES:
        token_tags[i] = 1  # TAG_TEXT
        timestep_indices[i] = 0
    for n, i in enumerate(AUDIO_INDICES):
        token_tags[i] = 2  # TAG_AUDIO
        timestep_indices[i] = n % 2
    for n, i in enumerate(VIDEO_INDICES):
        token_tags[i] = 0  # TAG_VIDEO
        timestep_indices[i] = n % 2

    position_ids = torch.zeros((SEQ_LEN, 3), dtype=torch.float32)
    for r in range(SEQ_LEN):
        position_ids[r, 0] = float(r)
        position_ids[r, 1] = float(r % 3)
        position_ids[r, 2] = float(r % 2)

    timestep = torch.tensor(TIMESTEPS, dtype=torch.float32)

    return dict(
        hidden_states=hidden_states,
        audio_hidden_states=audio_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=torch.tensor(VIDEO_INDICES, dtype=torch.long),
        audio_indices=torch.tensor(AUDIO_INDICES, dtype=torch.long),
        text_indices=torch.tensor(TEXT_INDICES, dtype=torch.long),
    )


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    manifest[name] = {"tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: {len(tensors)} tensors", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to the real MiniMax-H3 transformer/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    print(f"loading REAL MiniMaxH3Transformer3DModel from {args.checkpoint} (fp32, this is the ~66GB checkpoint - expect several minutes) ...", flush=True)
    model = MiniMaxH3Transformer3DModel.from_pretrained(args.checkpoint, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded: {n_params} params, num_layers={cfg.num_layers}, hidden_size={cfg.hidden_size}", flush=True)

    torch.manual_seed(args.seed)
    inputs = build_inputs(cfg, args.seed)

    mid = cfg.num_layers // 2
    last = cfg.num_layers - 1

    taps = {}

    def pre_hook(_mod, args_, _kwargs):
        taps["tap_block0_input"] = args_[0].detach().clone()

    def attn_hook(_mod, _args, out):
        taps["tap_block0_attn_out"] = out.detach().clone()

    def block_hook_factory(name):
        def hook(_mod, _args, out):
            taps[name] = out.detach().clone()
        return hook

    def temb_hook(_mod, _args, out):
        taps["tap_temb"] = out.detach().clone()

    def refiner_hook(_mod, _args, out):
        taps["tap_refiner_out"] = out.detach().clone()

    handles = [
        model.transformer_blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True),
        model.transformer_blocks[0].attn.register_forward_hook(attn_hook),
        model.transformer_blocks[0].register_forward_hook(block_hook_factory("tap_block0_out")),
        model.transformer_blocks[mid].register_forward_hook(block_hook_factory(f"tap_block{mid}_out")),
        model.transformer_blocks[last].register_forward_hook(block_hook_factory(f"tap_block{last}_out")),
        model.time_embedder.register_forward_hook(temb_hook),
        model.token_refiner.register_forward_hook(refiner_hook),
    ]

    print("running forward (small synthetic sequence, real weights) ...", flush=True)
    out = model(**inputs, return_dict=True)

    for h in handles:
        h.remove()

    cos_direct, sin_direct = model.rope(inputs["position_ids"])
    print(f"  self-validate rope: cos/sin shape {tuple(cos_direct.shape)}", flush=True)

    expect_taps = {"tap_block0_input", "tap_block0_attn_out", "tap_block0_out", f"tap_block{mid}_out", f"tap_block{last}_out", "tap_temb", "tap_refiner_out"}
    assert expect_taps == set(taps.keys()), f"tap set mismatch: got {set(taps.keys())}"

    tensors = {}
    for k in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        tensors[f"input_{k}"] = inputs[k][0]
    for k in ("timestep", "timestep_indices", "token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"):
        tensors[f"input_{k}"] = inputs[k]

    tensors["output_video"] = out.sample[0]
    tensors["output_audio"] = out.audio_sample[0]
    tensors["tap_rope_cos"] = cos_direct
    tensors["tap_rope_sin"] = sin_direct
    for k, v in taps.items():
        tensors[k] = v[0] if v.dim() == 3 and v.shape[0] == 1 else v

    manifest = {
        "run": {"seed": args.seed, "num_text": NUM_TEXT, "num_audio": NUM_AUDIO, "num_video": NUM_VIDEO, "seq_len": SEQ_LEN, "timesteps": TIMESTEPS, "mid_block": mid, "last_block": last},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    manifest["source"] = source_block(
        checkpoint="MiniMaxAI/MiniMax-H3",
        files=(),
        hash_files=False,
        identity={
            "num_attention_heads": cfg.num_attention_heads,
            "attention_head_dim": cfg.attention_head_dim,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "num_refiner_layers": cfg.num_refiner_layers,
            "ffn_dim": cfg.ffn_dim,
            "in_channels": cfg.in_channels,
            "audio_in_channels": cfg.audio_in_channels,
            "text_dim": cfg.text_dim,
            "freq_dim": cfg.freq_dim,
            "time_embed_hidden_dim": cfg.time_embed_hidden_dim,
            "time_embed_dim": cfg.time_embed_dim,
            "rope_freq_dim": cfg.rope_freq_dim,
            "rope_theta_x1000": int(cfg.rope_theta * 1000),
        },
    )
    save(args.out, "minimaxh3_dit_real.safetensors", tensors, manifest)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
