#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a single REAL decoder layer's forward from the actual
`Qwen/Qwen3.8-27B-FP8` checkpoint (M10's real-weight streaming parity
milestone) - the real-weight counterpart of `qwen35_dump_reference.py`
(tiny, random weights, whole text tower).

Unlike that dumper, this one never instantiates the whole 64-layer, ~108 GB
(dequantized) model - it builds exactly ONE standalone `Qwen3_5DecoderLayer`,
loads only that layer's own weights (dequantized from FP8 in-process, never
written to disk), and runs one forward on a fixed random input. Peak host
RAM stays at "one layer's weights + activations", matching the RAM
discipline `crates/qwen35/src/import.rs::import_layer` (the Rust side of
this same milestone) was written for.

Outputs (under `--out`, default `testdata/golden/qwen35/real_layer_{L}/`):
  layer.safetensors   x_in, out, cos, sin (all row 0 of batch=1)
  manifest.json        layer index, block_type, seed, T, sha256, versions

Usage:
  python tools/goldens/qwen35_dump_real_layer_reference.py \
      --dir /path/to/Qwen3.8-27B-FP8 --layer 0 [--out DIR] [--seed N] [--tokens T]
"""

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402


def dequant_block128(raw: torch.Tensor, scale_inv: torch.Tensor, block: int = 128) -> torch.Tensor:
    """`out[r,c] = raw[r,c] * scale_inv[r//block, c//block]` - byte-identical
    semantics to `model::fp8::dequant_block128` (Rust side), checked directly
    against it by `crates/qwen35/tests/real_weight_streaming.rs`."""
    rows, cols = raw.shape
    w = raw.to(torch.float32)
    s = scale_inv.to(torch.float32)
    s_full = s.repeat_interleave(block, dim=0)[:rows].repeat_interleave(block, dim=1)[:, :cols]
    return w * s_full


def load_layer_state_dict(shard_path: str, layer_idx: int) -> dict[str, torch.Tensor]:
    """Stream just `model.language_model.layers.{layer_idx}.*` out of one
    shard file, dequantizing every FP8 `.weight`/`.weight_scale_inv` pair -
    the same "one shard, one layer, dequantize in place" discipline
    `import_layer` (Rust) follows, so both sides load the identical bytes."""
    prefix = f"model.language_model.layers.{layer_idx}."
    sd: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt") as f:
        names = [k for k in f.keys() if k.startswith(prefix)]
        assert names, f"no tensors with prefix {prefix} in {shard_path}"
        for name in names:
            if name.endswith(".weight_scale_inv"):
                continue
            t = f.get_tensor(name)
            scale_name = f"{name}_scale_inv"
            if scale_name in names:
                t = dequant_block128(t, f.get_tensor(scale_name))
            else:
                t = t.to(torch.float32)
            local = name[len(prefix) :]
            sd[local] = t
    # Unlike brain's own import (`crate::qwen35::import::squeeze_conv1d`,
    # which drops the dead middle dim for its own kernel convention), the
    # REAL `nn.Conv1d.weight` parameter this loads into keeps the checkpoint's
    # own `[conv_dim, 1, kernel]` shape unchanged - no squeeze here.
    return sd


def save(out_dir, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out_dir, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    total = sum(v.numel() for v in tensors.values()) * 4 / 1e6
    print(f"wrote {name}: {len(tensors)} tensors, {total:.3f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="real checkpoint directory (config.json + layers-N.safetensors)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--tokens", type=int, default=12)
    args = ap.parse_args()
    out = args.out or os.path.join("testdata", "golden", "qwen35", f"real_layer_{args.layer}")
    os.makedirs(out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.dir).text_config
    block_type = cfg.layer_types[args.layer]
    print(f"layer {args.layer}: block_type={block_type}, hidden_size={cfg.hidden_size}")

    layer = Qwen3_5DecoderLayer(cfg, args.layer).eval()
    shard = os.path.join(args.dir, f"layers-{args.layer}.safetensors")
    sd = load_layer_state_dict(shard, args.layer)
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing, f"layer {args.layer}: missing keys after loading real weights: {missing}"
    assert not unexpected, f"layer {args.layer}: unexpected keys in real shard: {unexpected}"
    print(f"loaded {len(sd)} real tensors into layer {args.layer} ({block_type})")

    gen = torch.Generator().manual_seed(args.seed)
    T, b = args.tokens, 1
    x_in = torch.randn(b, T, cfg.hidden_size, generator=gen) * 0.02

    # M-RoPE tables: text-only degenerate case (all 3 axes equal), exactly
    # `qwen35_dump_reference.py`'s own convention.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    rotary_emb = Qwen3_5TextRotaryEmbedding(cfg)
    position_ids = torch.arange(T).view(1, 1, -1).expand(4, b, -1)
    rope_position_ids = position_ids[1:]
    cos, sin = rotary_emb(x_in, rope_position_ids)

    causal_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)[None, None, :, :]

    with torch.no_grad():
        out_hidden = layer(x_in, position_embeddings=(cos, sin), attention_mask=causal_mask)

    assert out_hidden.shape == x_in.shape
    assert torch.isfinite(out_hidden).all(), "real-layer forward produced a non-finite value"

    tensors = {"x_in": x_in[0], "out": out_hidden[0], "cos": cos[0], "sin": sin[0]}
    save(out, "layer.safetensors", tensors, manifest := {})
    manifest["_meta"] = {
        "layer": args.layer,
        "block_type": block_type,
        "seed": args.seed,
        "T": T,
        "B": b,
        "hidden_size": cfg.hidden_size,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "checkpoint_dir_basename": os.path.basename(os.path.normpath(args.dir)),
    }
    # `hash_files=False`: one 27B layer shard is GBs, and hashing it would
    # dominate a dump whose whole point is one layer. The identity carries the
    # enforced half regardless.
    manifest["source"] = source_block(
        checkpoint="Qwen/Qwen3.8-27B-FP8",
        files=[shard],
        hash_files=False,
        identity={
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "num_hidden_layers": cfg.num_hidden_layers,
            # The dumped tensors are ONE layer's, and which layer only fixes a
            # shape via its block type (GDN and GQA layers differ), so the
            # index is part of the identity, not just of `_meta`.
            "layer": args.layer,
        },
    )
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote manifest.json -> {out}")


if __name__ == "__main__":
    main()
