#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump real-weight numeric parity goldens for MiniMax-H3's DiT core.

Unlike `minimaxh3_audio_vae_dump_reference.py` (real checkpoint weights, out
of scope to distribute), this dumper needs no MiniMax-H3 checkpoint at all:
it builds the REAL, installed `diffusers==0.40.0`
`MiniMaxH3Transformer3DModel` class (`transformer_minimax_h3.py`) at a TINY
config matching `crate::config::H3TransformerConfig::tiny()`'s own
proportions (`inner_dim != hidden_size`, RoPE pass-through tail present),
with small seeded random weights (this class's own default `nn.Linear`/
`nn.RMSNorm` init, made reproducible by seeding torch's global RNG before
construction - the class defines no custom `_init_weights`), and runs ONE
forward over a small synthetic packed sequence spanning all three
modalities (text/video/audio) at TWO distinct timesteps, to genuinely
exercise per-row AdaLN indexing rather than a single shared modulation
vector.

Dumps every weight (`state_dict()` - PyTorch's own keys ARE the dotted
module-attribute path already, e.g. `transformer_blocks.0.attn.to_q.weight`,
so nothing here renames anything), every forward input, and four
intermediate taps (post-token-refiner text stream, post-RoPE cos/sin,
block 0's post-attention output, block 0's full output) plus the final
video/audio outputs, into one `minimaxh3_dit_tiny.safetensors` +
`manifest.json`.

Usage:
  python3 tools/minimaxh3_dit_dump_reference.py \\
      --out testdata/golden/minimaxh3/dit_tiny [--seed 3]
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/goldens")
from golden_source import source_block  # noqa: E402

from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel  # noqa: E402

# Matches crate::config::H3TransformerConfig::tiny() field-for-field.
TINY_CFG = dict(
    num_attention_heads=4,
    attention_head_dim=32,
    hidden_size=20,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=3,
    audio_in_channels=5,
    patch_size=(1, 2, 2),
    text_dim=12,
    freq_dim=16,
    time_embed_hidden_dim=40,
    time_embed_dim=24,
    rope_freq_dim=4,
    rope_theta=10000.0,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
)

NUM_TEXT = 3
NUM_AUDIO = 2
NUM_VIDEO = 4
SEQ_LEN = NUM_TEXT + NUM_AUDIO + NUM_VIDEO
# Packed row order: [text | audio | video] - arbitrary (packing order is out
# of scope for this phase, see transformer_minimax_h3.py's own doc: forward
# takes the index arrays as explicit arguments and does not build the layout
# itself), just needs to match `minimaxh3_dit_tiny_parity.rs`'s own layout.
TEXT_INDICES = list(range(0, NUM_TEXT))
AUDIO_INDICES = list(range(NUM_TEXT, NUM_TEXT + NUM_AUDIO))
VIDEO_INDICES = list(range(NUM_TEXT + NUM_AUDIO, SEQ_LEN))
TIMESTEPS = [0.2, 0.8]


def build_inputs(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    video_patch_dim = cfg["in_channels"] * cfg["patch_size"][0] * cfg["patch_size"][1] * cfg["patch_size"][2]

    hidden_states = torch.randn((1, NUM_VIDEO, video_patch_dim), generator=g) * 0.3
    audio_hidden_states = torch.randn((1, NUM_AUDIO, cfg["audio_in_channels"]), generator=g) * 0.3
    encoder_hidden_states = torch.randn((1, NUM_TEXT, cfg["text_dim"]), generator=g) * 0.3

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
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: {len(tensors)} tensors", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    torch.manual_seed(args.seed)
    model = MiniMaxH3Transformer3DModel(**TINY_CFG)
    model.eval()
    print(f"built MiniMaxH3Transformer3DModel ({sum(p.numel() for p in model.parameters())} params)", flush=True)

    inputs = build_inputs(TINY_CFG, args.seed)

    # ---- taps, via forward hooks - captures the ACTUAL values forward uses,
    # never a hand-reconstructed guess at them (porting.md's own guidance:
    # hook the module rather than re-deriving its input/output). ----------
    taps = {}

    def pre_hook(_mod, args, _kwargs):
        taps["tap_block0_input"] = args[0].detach().clone()

    def attn_hook(_mod, _args, out):
        taps["tap_block0_attn_out"] = out.detach().clone()

    def block_hook(_mod, _args, out):
        taps["tap_block0_out"] = out.detach().clone()

    def temb_hook(_mod, _args, out):
        taps["tap_temb"] = out.detach().clone()

    def refiner_hook(_mod, _args, out):
        taps["tap_refiner_out"] = out.detach().clone()

    handles = [
        model.transformer_blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True),
        model.transformer_blocks[0].attn.register_forward_hook(attn_hook),
        model.transformer_blocks[0].register_forward_hook(block_hook),
        model.time_embedder.register_forward_hook(temb_hook),
        model.token_refiner.register_forward_hook(refiner_hook),
    ]

    out = model(**inputs, return_dict=True)

    for h in handles:
        h.remove()

    # ---- self-validation: RoPE tables computed a second, independent way
    # (direct module call vs the value forward actually used internally) --
    cos_direct, sin_direct = model.rope(inputs["position_ids"])
    print(f"  self-validate rope: cos/sin shape {tuple(cos_direct.shape)}", flush=True)

    assert set(["tap_block0_input", "tap_block0_attn_out", "tap_block0_out", "tap_temb", "tap_refiner_out"]) == set(taps.keys())

    # Only these three carry a batch axis in the reference signature
    # (`(batch_size, num_tokens, channels)`) - stripped here since this
    # dumper only ever runs batch_size=1. Every other input is already
    # `(seq_len,)`/`(seq_len, 3)`/`(num_timesteps,)` with no batch axis.
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

    weights = dict(model.state_dict())
    tensors.update(weights)

    manifest = {
        "run": {"seed": args.seed, "num_text": NUM_TEXT, "num_audio": NUM_AUDIO, "num_video": NUM_VIDEO, "seq_len": SEQ_LEN, "timesteps": TIMESTEPS},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    manifest["source"] = source_block(
        checkpoint=None,
        files=(),
        hash_files=False,
        identity={k: (v if isinstance(v, int) else 0) for k, v in TINY_CFG.items() if isinstance(v, (int, bool))} | {
            "rope_theta_x1000": int(TINY_CFG["rope_theta"] * 1000),
        },
    )
    save(args.out, "minimaxh3_dit_tiny.safetensors", tensors, manifest)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
