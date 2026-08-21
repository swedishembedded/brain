#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a single REAL decoder layer's forward from MiniMax Music 3's
Global LLM checkpoint (`language_model/` - a genuine `Qwen3ForCausalLM`
re-export; see `crates/minimaxmusic3::global_llm::import`'s own doc for
why that directory and not the repository's other, differently-shaped
`qwen_7B/qwen_7B/` language-model directory).

Unlike this crate's other dumpers, this one needs no `diffusers` PR
dependency (the Global LLM is a real `Qwen3ForCausalLM`, covered by
plain `transformers>=4.51`, already in `requirements.txt`) - and unlike
`minimaxmusic3_dump_reference.py`'s four diffusers-class dumps, this one
never instantiates the whole 36-layer, ~17 GB model: it builds exactly
ONE standalone `Qwen3DecoderLayer`, loads only that layer's own weights
by resolving `model.safetensors.index.json`'s `weight_map` down to the
one (or two, for a layer split across a shard boundary) shard file(s)
that actually hold it, and runs one forward on a fixed random input.
Peak host RAM stays at "one layer's weights + activations", matching
`crates/qwen3::import::hf_source`'s own streaming discipline (the Rust
side of this same milestone) - and mirrors this repo's own
`qwen35_dump_real_layer_reference.py` precedent for the identical
"real weights, too big to load whole" situation.

Outputs (under `--out`, default `testdata/golden/minimaxmusic3/
global_llm_layer_{L}/`):
  layer.safetensors   x_in, out, cos, sin (all row 0 of batch=1)
  manifest.json        layer index, seed, T, sha256, versions

Usage:
  python tools/goldens/minimaxmusic3_global_llm_dump_reference.py \
      --dir /path/to/language_model --layer 0 [--out DIR] [--seed N] [--tokens T]
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
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402


def load_layer_state_dict(checkpoint_dir: str, layer_idx: int) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Resolve `model.safetensors.index.json`'s `weight_map` down to just
    `model.layers.{layer_idx}.*`, open only the shard file(s) that own
    those names, and return the plain-fp32 state dict a bare
    `Qwen3DecoderLayer` (whose own parameter names have no `model.layers.
    {layer_idx}.` prefix) can load directly, plus the shard file basenames
    actually read (for the golden manifest's own provenance record)."""
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"model.layers.{layer_idx}."
    names = [n for n in weight_map if n.startswith(prefix)]
    assert names, f"no tensors with prefix {prefix} in {index_path}"

    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)

    sd: dict[str, torch.Tensor] = {}
    for shard_file, shard_names in by_shard.items():
        with safe_open(os.path.join(checkpoint_dir, shard_file), framework="pt") as f:
            for name in shard_names:
                sd[name[len(prefix) :]] = f.get_tensor(name).to(torch.float32)
    return sd, sorted(by_shard.keys())


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
    ap.add_argument("--dir", required=True, help="the checkpoint's language_model/ directory (config.json + model-*-of-*.safetensors)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--tokens", type=int, default=12)
    args = ap.parse_args()
    out = args.out or os.path.join("testdata", "golden", "minimaxmusic3", f"global_llm_layer_{args.layer}")
    os.makedirs(out, exist_ok=True)

    cfg = Qwen3Config.from_pretrained(args.dir)
    assert cfg.model_type == "qwen3", f"expected a genuine Qwen3 config, got model_type={cfg.model_type!r} (wrong directory?)"
    print(f"layer {args.layer}: hidden_size={cfg.hidden_size}, num_attention_heads={cfg.num_attention_heads}, num_key_value_heads={cfg.num_key_value_heads}, head_dim={cfg.head_dim}")

    layer = Qwen3DecoderLayer(cfg, args.layer).eval()
    sd, shard_files = load_layer_state_dict(args.dir, args.layer)
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing, f"layer {args.layer}: missing keys after loading real weights: {missing}"
    assert not unexpected, f"layer {args.layer}: unexpected keys in real shard: {unexpected}"
    print(f"loaded {len(sd)} real tensors into layer {args.layer}")

    gen = torch.Generator().manual_seed(args.seed)
    T, b = args.tokens, 1
    x_in = torch.randn(b, T, cfg.hidden_size, generator=gen) * 0.02

    rotary_emb = Qwen3RotaryEmbedding(cfg)
    position_ids = torch.arange(T).view(1, -1)
    cos, sin = rotary_emb(x_in, position_ids)

    causal_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)[None, None, :, :]

    with torch.no_grad():
        out_hidden = layer(x_in, attention_mask=causal_mask, position_ids=position_ids, position_embeddings=(cos, sin))

    assert out_hidden.shape == x_in.shape
    assert torch.isfinite(out_hidden).all(), "real-layer forward produced a non-finite value"

    tensors = {"x_in": x_in[0], "out": out_hidden[0], "cos": cos[0], "sin": sin[0]}
    save(out, "layer.safetensors", tensors, manifest := {})
    manifest["_meta"] = {
        "layer": args.layer,
        "seed": args.seed,
        "T": T,
        "B": b,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "rope_theta": getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"],
        "rms_norm_eps": cfg.rms_norm_eps,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "checkpoint_dir_basename": os.path.basename(os.path.normpath(args.dir)),
    }
    # Top level, per brain_testutil::golden::Source::open_manifest's own
    # contract (it reads manifest["source"], not a nested key). Multi-GB
    # shards: hashing them would dominate this dumper's runtime for no real
    # benefit (the layer/shape identity below is what a parity test
    # actually checks against) - see source_block's own doc.
    manifest["source"] = source_block(
        checkpoint="MiniMaxAI/MiniMax-Music3",
        files=shard_files,
        hash_files=False,
        identity={
            "layer": args.layer,
            "hidden_size": int(cfg.hidden_size),
            "num_attention_heads": int(cfg.num_attention_heads),
            "num_key_value_heads": int(cfg.num_key_value_heads),
            "head_dim": int(cfg.head_dim),
        },
    )
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote manifest.json -> {out}")


if __name__ == "__main__":
    main()
