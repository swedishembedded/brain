#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving that ported hybrid
# local/global attention decoders reproduce their reference models for its
# clients. If your team needs expertise in layer-by-layer parity validation of
# large language models then you can procure our services by sending an email
# to info@swedishembedded.com.

"""Dump Ripple (`RicursiveForCausalLM`) reference goldens.

Runs the OFFICIAL `modeling_ricursive.py` shipped with the checkpoint (CPU,
fp32), imported from a directory given on the command line, and dumps every
boundary a Rust replay needs. Two modes, each writing its own directory with a
`manifest.json`:

  tiny   random weights, a 3-layer model (sliding, sliding, full) whose window
         (4) is smaller than the prompt (12) so it excludes keys, with a
         nonzero `value_res_lambda` per layer and an FFN clamp threshold (0.5)
         small enough that both the gate cap and the up clip fire. Writes
           <out>/tiny/ricursive_tiny.safetensors          taps
           <out>/tiny/ricursive_tiny_weights.safetensors  the model's weights
           <out>/tiny/config.json                         its config
  real   the truncated real model: layers 0 (sliding) and 4 (full) of the
         released checkpoint, run as a 2-layer model with
         `layer_types = [sliding, full]` and the real width, embed, final norm
         and `lm_head`. bf16 weights are promoted to fp32. Writes
           <out>/real/ricursive_real_reduced.safetensors  taps

Taps (all float32; token ids are cast exactly): `input_ids`, `hidden.0` (the
scaled embeddings), `hidden.{k+1}` (decoder layer k's raw output),
`attn_out.{k}` (layer k's `o_proj` output, before `post_attention_layernorm`),
`final_hidden` (after the final norm) and `logits`.

The reference imports `.configuration_ricursive` relatively, so the directory
is registered as a package. Run with `python -I`; nothing is read from the
current directory.

Usage:
  python -I ricursive_dump_reference.py tiny --modeling-dir DIR --out DIR
  python -I ricursive_dump_reference.py real --modeling-dir DIR --out DIR \\
      --checkpoint layers-0-4.safetensors --config config.json \\
      --tokenizer tokenizer.json
"""

import argparse
import hashlib
import importlib
import json
import os
import secrets
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

# Real layers 0 and 4 become layers 0 and 1 of the truncated reference.
REAL_SOURCE_LAYERS = (0, 4)
REAL_PROMPT = "The capital of Sweden is Stockholm, and the capital of Norway is"

TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=256,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    sliding_window=4,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"],
    use_gated_attention=True,
    gate_full_precision=True,
    use_valres=True,
    ffn_clamp_threshold=0.5,
    embed_scale=8.0,  # sqrt(hidden_size), the released checkpoint's convention
    tie_word_embeddings=False,
)
TINY_PROMPT_LEN = 12


def load_reference(modeling_dir):
    """Import modeling_ricursive.py as `ricursive_ref.modeling_ricursive`."""
    pkg = types.ModuleType("ricursive_ref")
    pkg.__path__ = [modeling_dir]
    sys.modules["ricursive_ref"] = pkg
    return (
        importlib.import_module("ricursive_ref.configuration_ricursive"),
        importlib.import_module("ricursive_ref.modeling_ricursive"),
    )


class Taps:
    """Forward hooks recording each layer's output, its attention output and
    the two pre-activation FFN branches (for the clamp-firing self-check)."""

    def __init__(self, model):
        self.layer_out = {}
        self.attn_out = {}
        self.gate = {}
        self.up = {}
        for k, layer in enumerate(model.model.layers):
            layer.register_forward_hook(self._save(self.layer_out, k, first=True))
            layer.self_attn.register_forward_hook(self._save(self.attn_out, k, first=True))
            layer.mlp.gate_proj.register_forward_hook(self._save(self.gate, k))
            layer.mlp.up_proj.register_forward_hook(self._save(self.up, k))

    @staticmethod
    def _save(store, k, first=False):
        def hook(_module, _args, out):
            store[k] = (out[0] if first else out).detach().clone()
        return hook


def run(model, ids):
    taps = Taps(model)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    n = len(model.model.layers)
    tensors = {"input_ids": ids[0].to(torch.float32), "hidden.0": out.hidden_states[0][0]}
    for k in range(n):
        tensors[f"hidden.{k + 1}"] = taps.layer_out[k][0]
        tensors[f"attn_out.{k}"] = taps.attn_out[k][0]
    tensors["final_hidden"] = out.hidden_states[-1][0]
    tensors["logits"] = out.logits[0]
    return tensors, taps, out.logits


def save(path, tensors):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    save_file(tensors, path)
    mb = sum(v.numel() for v in tensors.values()) * 4 / 1e6
    print(f"wrote {path}: {len(tensors)} tensors, {mb:.2f} MB", flush=True)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def tiny(args, cfg_mod, mdl_mod):
    torch.manual_seed(args.seed)
    cfg = cfg_mod.RicursiveConfig(**TINY)
    model = mdl_mod.RicursiveForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("value_res_lambda"):
                p.copy_(torch.randn(1) * 1.2)
            elif p.dim() == 1:  # norm gains: around 1, never exactly 1
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(0.35 * torch.randn_like(p))
    ids = torch.randint(0, TINY["vocab_size"], (1, TINY_PROMPT_LEN))

    tensors, taps, logits = run(model, ids)

    # Self-checks: the golden must exercise what it claims to.
    for k in range(TINY["num_hidden_layers"]):
        n_gate = (taps.gate[k] > TINY["ffn_clamp_threshold"]).sum().item()
        n_up = (taps.up[k].abs() > TINY["ffn_clamp_threshold"]).sum().item()
        print(f"  layer {k}: gate cap fires on {n_gate}, up clip on {n_up} elements", flush=True)
        assert n_gate > 0 and n_up > 0, f"layer {k}: a clamp never fires"
    def logits_with(**overrides):
        variant = mdl_mod.RicursiveForCausalLM(cfg_mod.RicursiveConfig(**{**TINY, **overrides})).float().eval()
        variant.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            return variant(input_ids=ids, use_cache=False).logits

    window_effect = (logits_with(sliding_window=1 << 20) - logits).abs().max().item()
    print(f"  window changes logits by up to {window_effect:.3e}", flush=True)
    assert window_effect > 1e-3, "the sliding window does not exclude anything"
    clamp_effect = (logits_with(ffn_clamp_threshold=None) - logits).abs().max().item()
    print(f"  clamp changes logits by up to {clamp_effect:.3e}", flush=True)
    assert clamp_effect > 1e-3, "the FFN clamp does not change the logits"

    out = os.path.join(args.out, "tiny")
    os.makedirs(out, exist_ok=True)
    manifest = {}
    manifest["ricursive_tiny.safetensors"] = save(os.path.join(out, "ricursive_tiny.safetensors"), tensors)
    weights = {k: v for k, v in model.state_dict().items()}
    manifest["ricursive_tiny_weights.safetensors"] = save(os.path.join(out, "ricursive_tiny_weights.safetensors"), weights)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=True)
    finish(out, manifest, TINY, args, checkpoint=None, files=[])


def real(args, cfg_mod, mdl_mod):
    with open(args.config) as f:
        full = json.load(f)
    layer_types = [full["layer_types"][i] for i in REAL_SOURCE_LAYERS]
    assert layer_types == ["sliding_attention", "full_attention"], layer_types
    keys = [
        "vocab_size", "hidden_size", "intermediate_size", "num_attention_heads",
        "num_key_value_heads", "head_dim", "max_position_embeddings", "hidden_act",
        "rms_norm_eps", "rope_theta", "sliding_window", "use_gated_attention",
        "gate_full_precision", "use_valres", "ffn_clamp_threshold", "embed_scale",
        "tie_word_embeddings",
    ]
    cfg = cfg_mod.RicursiveConfig(
        num_hidden_layers=len(REAL_SOURCE_LAYERS), layer_types=layer_types, **{k: full[k] for k in keys}
    )
    with torch.device("meta"):
        model = mdl_mod.RicursiveForCausalLM(cfg)
    rename = {f"model.layers.{src}.": f"model.layers.{dst}." for dst, src in enumerate(REAL_SOURCE_LAYERS)}
    state = {}
    with safe_open(args.checkpoint, "pt") as f:
        for name in f.keys():
            new = name
            for old, repl in rename.items():
                if name.startswith(old):
                    new = repl + name[len(old):]
            state[new] = f.get_tensor(name).to(torch.float32)
    model.load_state_dict(state, strict=True, assign=True)  # refuses missing and unexpected keys
    model = model.float().eval()

    from tokenizers import Tokenizer

    ids = torch.tensor([Tokenizer.from_file(args.tokenizer).encode(REAL_PROMPT).ids])
    print(f"prompt ids ({ids.shape[1]}): {ids[0].tolist()}", flush=True)
    tensors, _taps, _logits = run(model, ids)

    out = os.path.join(args.out, "real")
    os.makedirs(out, exist_ok=True)
    manifest = {"ricursive_real_reduced.safetensors": save(os.path.join(out, "ricursive_real_reduced.safetensors"), tensors)}
    identity = dict(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers, num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim, sliding_window=cfg.sliding_window,
    )
    finish(out, manifest, identity, args, checkpoint="ricursiveAI/Ripple-65B-Preview", files=[args.checkpoint])


def finish(out, manifest, identity, args, checkpoint, files):
    ident = {k: v for k, v in identity.items() if isinstance(v, int) and not isinstance(v, bool)}
    manifest["source"] = source_block(identity=ident, checkpoint=checkpoint, files=files, hash_files=False)
    manifest["seed"] = args.seed
    manifest["versions"] = {"torch": torch.__version__, "python": sys.version.split()[0]}
    try:
        import transformers
        manifest["versions"]["transformers"] = transformers.__version__
    except ImportError:
        pass
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {out}/manifest.json", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tiny", "real"])
    ap.add_argument("--modeling-dir", required=True, help="directory holding modeling_ricursive.py and configuration_ricursive.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", help="real: the partial safetensors holding embed, norm, lm_head and layers 0 and 4")
    ap.add_argument("--config", help="real: the released config.json")
    ap.add_argument("--tokenizer", help="real: the released tokenizer.json")
    ap.add_argument("--seed", type=int, default=None, help="tiny: weight and prompt seed (default: random, printed and recorded in the manifest)")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(31)
    print(f"seed {args.seed}", flush=True)
    cfg_mod, mdl_mod = load_reference(os.path.abspath(args.modeling_dir))
    if args.mode == "real" and not (args.checkpoint and args.config and args.tokenizer):
        ap.error("real mode needs --checkpoint, --config and --tokenizer")
    {"tiny": tiny, "real": real}[args.mode](args, cfg_mod, mdl_mod)


if __name__ == "__main__":
    main()
