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

  grads  the same tiny model, trained one batch: the loss, the gradient of
         every parameter, and a few steps of `torch.optim.AdamW` with global
         gradient-norm clipping. Writes
           <out>/tiny_train/ricursive_tiny_train.safetensors          the batch, loss, `grad.<name>`, the loss
                                                                      of each AdamW step and `final.<name>`
           <out>/tiny_train/ricursive_tiny_train_weights.safetensors  the model's initial weights
           <out>/tiny_train/config.json                               its config
  real-lora  the truncated real model of `real` with a rank-8 LoRA adapter on every linear of both layers
         (q, k, v, o, g, gate, up, down), the adapter's A and B seeded here and dumped with the batch: the loss,
         the gradient of every A and B in full, one `torch.optim.AdamW` step on them (the base frozen) and the
         loss and factors after it. Writes
           <out>/real_lora/ricursive_real_lora.safetensors
  real-grads  the truncated real model of `real`, one backward on a short
         prompt: the loss and, per parameter, the gradient's L2 norm and
         4096 evenly strided entries (the full gradients are 10 GB). Writes
           <out>/real_grads/ricursive_real_grads.safetensors

  full   the WHOLE released model (80 layers, 130 GB in bf16), run layer by layer: the embedding, then for each
         layer the official `RicursiveDecoderLayer` is built, loaded from the sharded checkpoint directory, applied
         to the residual stream and dropped, then the final norm and `lm_head`. Only one layer's weights (1.5 GB
         in bf16) are ever resident beside the stream, so a single card or a CPU reproduces the official forward
         without holding the model; the arithmetic is `RicursiveModel.forward`'s, layer for layer. Writes
           <out>/full/ricursive_full.safetensors  `input_ids`, `logits` (the last `--logits-rows` rows),
                                                  `hidden.<k>` (the stream entering layer k; the last `--tap-rows`
                                                  rows) for the layers of `--taps`, and `final_hidden`
  full-check  the streaming path against the whole-model path on a seeded tiny model written as a sharded
         checkpoint: the streamed logits and taps equal `run()`'s, each tensor is read exactly once and one layer
         is ever resident. Needs no checkpoint; this is the gate for `full` at tiny scale.

Taps (all float32; token ids are cast exactly): `input_ids`, `hidden.0` (the
scaled embeddings), `hidden.{k+1}` (decoder layer k's raw output),
`attn_out.{k}` (layer k's `o_proj` output, before `post_attention_layernorm`),
`final_hidden` (after the final norm) and `logits`.

The reference imports `.configuration_ricursive` relatively, so the directory
is registered as a package. Run with `python -I`; nothing is read from the
current directory.

Usage:
  python -I ricursive_dump_reference.py full --modeling-dir DIR --out DIR --checkpoint-dir RELEASE \\
      --tokenizer tokenizer.json --prompt-file text.txt --max-tokens 4096 --device cuda:0 --dtype bfloat16
  python -I ricursive_dump_reference.py full-check --modeling-dir DIR --out DIR
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
import struct
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


def tiny_model(args, cfg_mod, mdl_mod):
    """The random-weight tiny reference model of `tiny` and `grads`, built from the seed in `args`."""
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
    return cfg, model


def tiny(args, cfg_mod, mdl_mod):
    cfg, model = tiny_model(args, cfg_mod, mdl_mod)
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


# AdamW hyper-parameters and clip of the trajectory gate; the Rust test reads them back from the manifest.
TRAIN = dict(lr=3e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, clip=1.0, steps=3)


def training_batch(ids):
    """Next-token targets for `ids` (1, T): the last position and position 3 are unsupervised (-100)."""
    targets = torch.cat([ids[0, 1:], torch.tensor([-100])])
    targets[3] = -100
    return targets


def loss_of(model, ids, targets):
    """Mean cross-entropy over the supervised positions, as `F.cross_entropy(..., ignore_index=-100)`."""
    logits = model(input_ids=ids, use_cache=False).logits[0]
    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)


def grads(args, cfg_mod, mdl_mod):
    cfg, model = tiny_model(args, cfg_mod, mdl_mod)
    ids = torch.randint(0, TINY["vocab_size"], (1, TINY_PROMPT_LEN))
    targets = training_batch(ids)
    model.train()  # the reference has no dropout (attention_dropout is refused at 0); train() only enables autograd hooks

    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss = loss_of(model, ids, targets)
    loss.backward()
    tensors = {"input_ids": ids[0].to(torch.float32), "targets": targets.to(torch.float32), "loss": loss.detach().reshape(1)}
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        tensors[f"grad.{name}"] = p.grad.detach().clone()
    n_dead = [n for n, p in model.named_parameters() if p.grad.abs().max().item() == 0.0]
    assert not n_dead, f"parameters with an all-zero gradient: {n_dead}"

    # AdamW trajectory: the same batch, a few steps, clip by global norm first (torch's clip_grad_norm_).
    model.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], betas=(TRAIN["beta1"], TRAIN["beta2"]), eps=TRAIN["eps"], weight_decay=TRAIN["weight_decay"])
    losses = []
    for _ in range(TRAIN["steps"]):
        opt.zero_grad(set_to_none=True)
        step_loss = loss_of(model, ids, targets)
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["clip"])
        opt.step()
        losses.append(step_loss.item())
    tensors["trajectory.loss"] = torch.tensor(losses)
    tensors["trajectory.loss_after"] = loss_of(model, ids, targets).detach().reshape(1)
    for name, p in model.named_parameters():
        tensors[f"final.{name}"] = p.detach().clone()
    print(f"  loss {loss.item():.6f}, AdamW losses {losses}, after {tensors['trajectory.loss_after'].item():.6f}", flush=True)

    out = os.path.join(args.out, "tiny_train")
    os.makedirs(out, exist_ok=True)
    manifest = {}
    manifest["ricursive_tiny_train.safetensors"] = save(os.path.join(out, "ricursive_tiny_train.safetensors"), tensors)
    manifest["ricursive_tiny_train_weights.safetensors"] = save(os.path.join(out, "ricursive_tiny_train_weights.safetensors"), initial)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=True)
    manifest["train"] = TRAIN
    finish(out, manifest, TINY, args, checkpoint=None, files=[])


def real_model(args, cfg_mod, mdl_mod):
    """The truncated real model of `real` (layers 0 and 4), fp32 on CPU, ready to differentiate."""
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
    model.load_state_dict(state, strict=True, assign=True)
    return cfg, model.float()


def real_grads(args, cfg_mod, mdl_mod):
    from tokenizers import Tokenizer

    _cfg, model = real_model(args, cfg_mod, mdl_mod)
    ids = torch.tensor([Tokenizer.from_file(args.tokenizer).encode(REAL_PROMPT).ids])
    targets = training_batch(ids)
    loss = loss_of(model, ids, targets)
    loss.backward()
    tensors = {"input_ids": ids[0].to(torch.float32), "targets": targets.to(torch.float32), "loss": loss.detach().reshape(1)}
    for name, p in model.named_parameters():
        flat = p.grad.detach().reshape(-1)
        stride = max(1, flat.numel() // 4096)
        tensors[f"norm.{name}"] = flat.double().norm().float().reshape(1)
        tensors[f"sample.{name}"] = flat[::stride][:4096].clone()
    print(f"  prompt {ids.shape[1]} tokens, loss {loss.item():.6f}", flush=True)
    out = os.path.join(args.out, "real_grads")
    os.makedirs(out, exist_ok=True)
    manifest = {"ricursive_real_grads.safetensors": save(os.path.join(out, "ricursive_real_grads.safetensors"), tensors)}
    identity = dict(
        vocab_size=model.config.vocab_size, hidden_size=model.config.hidden_size, intermediate_size=model.config.intermediate_size,
        num_hidden_layers=model.config.num_hidden_layers, num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim, sliding_window=model.config.sliding_window,
    )
    finish(out, manifest, identity, args, checkpoint="ricursiveAI/Ripple-65B-Preview", files=[args.checkpoint])



# The adapter of the `real-lora` gate: rank, alpha (scale alpha / rank) and the AdamW step; its factors are drawn from --seed
# (random unless given, recorded in the manifest) and dumped, the Rust test reads them back.
LORA = dict(rank=8, alpha=16.0, lr=1e-3)
LORA_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "gate_proj", "up_proj", "down_proj")


class LoraLinear(torch.nn.Module):
    """`base(x) + alpha / rank * (x A^T) B^T`, the base frozen."""

    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base, self.scale = base, scale
        self.a, self.b = torch.nn.Parameter(a), torch.nn.Parameter(b)

    def forward(self, x):
        h = torch.nn.functional.linear(x, self.a)
        return self.base(x) + self.scale * torch.nn.functional.linear(h, self.b)


def real_lora(args, cfg_mod, mdl_mod):
    from tokenizers import Tokenizer

    _cfg, model = real_model(args, cfg_mod, mdl_mod)
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(args.seed)
    scale = LORA["alpha"] / LORA["rank"]
    sites = []  # (canonical weight name, module owner, leaf)
    for li, layer in enumerate(model.model.layers):
        for leaf in LORA_LEAVES:
            owner = layer.mlp if leaf.endswith(("gate_proj", "up_proj", "down_proj")) else layer.self_attn
            base = getattr(owner, leaf)
            out_f, in_f = base.weight.shape
            # A has the scale of a trained adapter's, B is nonzero (a zero B would leave A without a gradient).
            a = torch.randn(LORA["rank"], in_f, generator=gen) / in_f ** 0.5
            b = 0.02 * torch.randn(out_f, LORA["rank"], generator=gen)
            setattr(owner, leaf, LoraLinear(base, a, b, scale))
            group = "mlp" if owner is layer.mlp else "self_attn"
            sites.append((f"model.layers.{li}.{group}.{leaf}.weight", owner, leaf))
    ids = torch.tensor([Tokenizer.from_file(args.tokenizer).encode(REAL_PROMPT).ids])
    targets = training_batch(ids)
    tensors = {"input_ids": ids[0].to(torch.float32), "targets": targets.to(torch.float32)}
    for name, owner, leaf in sites:
        m = getattr(owner, leaf)
        tensors[f"a.{name}"], tensors[f"b.{name}"] = m.a.detach().clone(), m.b.detach().clone()
    params = [p for n, o, l in sites for p in (getattr(o, l).a, getattr(o, l).b)]
    loss = loss_of(model, ids, targets)
    loss.backward()
    tensors["loss"] = loss.detach().reshape(1)
    for name, owner, leaf in sites:
        m = getattr(owner, leaf)
        assert m.a.grad is not None and m.b.grad is not None and m.a.grad.abs().max().item() > 0, f"{name}: dead adapter gradient"
        tensors[f"grad_a.{name}"], tensors[f"grad_b.{name}"] = m.a.grad.detach().clone(), m.b.grad.detach().clone()
    opt = torch.optim.AdamW(params, lr=LORA["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    for name, owner, leaf in sites:
        m = getattr(owner, leaf)
        tensors[f"after_a.{name}"], tensors[f"after_b.{name}"] = m.a.detach().clone(), m.b.detach().clone()
    tensors["loss_after"] = loss_of(model, ids, targets).detach().reshape(1)
    print(f"  prompt {ids.shape[1]} tokens, loss {loss.item():.6f} -> {tensors['loss_after'].item():.6f} after one AdamW step", flush=True)
    out = os.path.join(args.out, "real_lora")
    os.makedirs(out, exist_ok=True)
    manifest = {"ricursive_real_lora.safetensors": save(os.path.join(out, "ricursive_real_lora.safetensors"), tensors), "lora": LORA}
    identity = dict(
        vocab_size=model.config.vocab_size, hidden_size=model.config.hidden_size, intermediate_size=model.config.intermediate_size,
        num_hidden_layers=model.config.num_hidden_layers, num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim, sliding_window=model.config.sliding_window,
    )
    finish(out, manifest, identity, args, checkpoint="ricursiveAI/Ripple-65B-Preview", files=[args.checkpoint])


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


class StreamedCheckpoint:
    """Tensors of a Hugging Face safetensors directory (sharded with `model.safetensors.index.json`, or one
    `model.safetensors`) read one at a time, never the whole model. Counts the reads of each tensor."""

    def __init__(self, root):
        self.root = root
        index = os.path.join(root, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.shard_of = json.load(f)["weight_map"]
        else:
            single = "model.safetensors"
            with safe_open(os.path.join(root, single), "pt") as f:
                self.shard_of = {name: single for name in f.keys()}
        self.handles = {}
        self.reads = {}

    def get(self, name):
        shard = self.shard_of[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.root, shard), "pt")
        self.reads[name] = self.reads.get(name, 0) + 1
        return self.handles[shard].get_tensor(name)

    def names_under(self, prefix):
        return sorted(n for n in self.shard_of if n.startswith(prefix))


class HttpCheckpoint:
    """[`StreamedCheckpoint`]'s interface over a Hugging Face repository read by HTTP range requests: each tensor
    is fetched into memory when asked for and never written to disk, so the 130 GB bf16 release is run on a drive
    that cannot hold it (the bytes cross the network once per tensor). `base` is the `.../resolve/main/` URL.
    Public repositories need no token; `HF_TOKEN`, when set, is sent."""

    DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "U8": torch.uint8, "I8": torch.int8}

    def __init__(self, base):
        self.base = base if base.endswith("/") else base + "/"
        self.shard_of = json.loads(self._range("model.safetensors.index.json", None))["weight_map"]
        self.headers = {}
        self.reads = {}

    def _range(self, name, span):
        import urllib.request

        request = urllib.request.Request(self.base + name)
        token = os.environ.get("HF_TOKEN")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        if span is not None:
            request.add_header("Range", f"bytes={span[0]}-{span[1] - 1}")
        with urllib.request.urlopen(request, timeout=600) as reply:
            data = reply.read()
        if span is not None and len(data) != span[1] - span[0]:
            raise SystemExit(f"{name} bytes {span}: the server answered {len(data)} bytes (no range support?)")
        return data

    def _header(self, shard):
        if shard not in self.headers:
            (n,) = struct.unpack("<Q", self._range(shard, (0, 8)))
            self.headers[shard] = (json.loads(self._range(shard, (8, 8 + n))), 8 + n)
        return self.headers[shard]

    def get(self, name):
        shard = self.shard_of[name]
        header, start = self._header(shard)
        entry = header[name]
        a, b = entry["data_offsets"]
        self.reads[name] = self.reads.get(name, 0) + 1
        data = bytearray(self._range(shard, (start + a, start + b)))
        return torch.frombuffer(data, dtype=self.DTYPES[entry["dtype"]]).reshape(entry["shape"])

    def names_under(self, prefix):
        return sorted(n for n in self.shard_of if n.startswith(prefix))


def open_checkpoint(location):
    """A directory path, or an `http(s)://` base URL (a Hugging Face `.../resolve/main/`)."""
    return HttpCheckpoint(location) if location.startswith(("http://", "https://")) else StreamedCheckpoint(location)


def streamed_forward(ckpt, cfg, mdl_mod, ids, device, dtype, tap_layers, tap_rows, logits_rows):
    """`RicursiveModel.forward` followed by the head, with one decoder layer resident at a time.

    Returns the tensors to dump and the statistics (`max_resident_layers`, `bytes_loaded`) that show nothing larger
    than a layer was held. The order of operations is the model's: scaled embeddings seed both the stream and the
    value residual, every layer sees `attention_mask=None` (the SDPA causal path), then the final norm, `lm_head`.
    """
    t = ids.shape[1]
    stats = {"max_resident_layers": 0, "bytes_loaded": 0}
    out = {"input_ids": ids[0].to(torch.float32).cpu()}

    def load(module, state):
        stats["bytes_loaded"] += sum(v.numel() * v.element_size() for v in state.values())
        module.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in state.items()}, strict=True, assign=True)
        return module

    with torch.inference_mode():
        table = ckpt.get("model.embed_tokens.weight")
        stats["bytes_loaded"] += table.numel() * table.element_size()
        stream = torch.nn.functional.embedding(ids.to(device), table.to(device=device, dtype=dtype))
        del table
        if getattr(cfg, "embed_scale", None) is not None:
            stream = stream * cfg.embed_scale
        value_residual = stream if cfg.use_valres else None
        position_ids = torch.arange(t, device=device).unsqueeze(0)
        for k in range(cfg.num_hidden_layers):
            if k in tap_layers:
                out[f"hidden.{k}"] = stream[0, -tap_rows:].float().cpu()
            prefix = f"model.layers.{k}."
            with torch.device("meta"):
                layer = mdl_mod.RicursiveDecoderLayer(cfg, k)
            state = {n[len(prefix):]: ckpt.get(n) for n in ckpt.names_under(prefix)}
            load(layer, state)
            del state
            stats["max_resident_layers"] = max(stats["max_resident_layers"], 1)
            stream = layer(
                stream, attention_mask=None, position_ids=position_ids, past_key_value=None, output_attentions=False,
                use_cache=False, cache_position=position_ids[0], value_residual=value_residual,
            )[0]
            del layer
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  layer {k + 1}/{cfg.num_hidden_layers}", flush=True)
        if cfg.num_hidden_layers in tap_layers:
            out[f"hidden.{cfg.num_hidden_layers}"] = stream[0, -tap_rows:].float().cpu()
        with torch.device("meta"):
            norm = mdl_mod.RicursiveRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        load(norm, {"weight": ckpt.get("model.norm.weight")})
        final = norm(stream)
        out["final_hidden"] = final[0, -tap_rows:].float().cpu()
        head = ckpt.get("lm_head.weight").to(device=device, dtype=dtype)
        stats["bytes_loaded"] += head.numel() * head.element_size()
        out["logits"] = torch.nn.functional.linear(final[0, -logits_rows:], head).float().cpu()
    return out, stats


def parse_taps(text, n_layers):
    """`every`, `none`, or a comma list of layer indices in 0..=n_layers (`n_layers` is the stream entering the norm)."""
    if text == "every":
        return set(range(n_layers + 1))
    if text == "none":
        return set()
    taps = {int(v) for v in text.split(",")}
    bad = sorted(v for v in taps if not 0 <= v <= n_layers)
    if bad:
        raise SystemExit(f"--taps {bad}: layers run 0 to {n_layers}")
    return taps


def full(args, cfg_mod, mdl_mod):
    if not args.checkpoint_dir:
        raise SystemExit("full needs --checkpoint-dir")
    if bool(args.prompt_file) == bool(args.ids_file):
        raise SystemExit("full needs exactly one of --prompt-file and --ids-file")
    if args.prompt_file and not args.tokenizer:
        raise SystemExit("--prompt-file needs --tokenizer")
    ckpt = open_checkpoint(args.checkpoint_dir)
    if isinstance(ckpt, HttpCheckpoint):
        raw = json.loads(ckpt._range("config.json", None))
    else:
        with open(os.path.join(args.checkpoint_dir, "config.json")) as f:
            raw = json.load(f)
    cfg = cfg_mod.RicursiveConfig.from_dict(raw)
    if args.ids_file:
        with open(args.ids_file) as f:
            token_ids = json.load(f)
    else:
        from tokenizers import Tokenizer

        with open(args.prompt_file, encoding="utf-8") as f:
            token_ids = Tokenizer.from_file(args.tokenizer).encode(f.read()).ids
    token_ids = token_ids[: args.max_tokens]
    ids = torch.tensor([token_ids])
    print(f"prompt: {ids.shape[1]} tokens; {cfg.num_hidden_layers} layers streamed on {args.device} in {args.dtype}", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tap_layers = parse_taps(args.taps, cfg.num_hidden_layers)
    tensors, stats = streamed_forward(ckpt, cfg, mdl_mod, ids, args.device, dtype, tap_layers, args.tap_rows, args.logits_rows)
    missing = sorted(n for n in ckpt.shard_of if n not in ckpt.reads)
    if missing:
        raise SystemExit(f"the checkpoint holds tensors the reference never read (a layer count mismatch?): {missing[:3]}")
    print(f"streamed {stats['bytes_loaded'] / 2**30:.1f} GiB through one resident layer", flush=True)

    out = os.path.join(args.out, "full")
    os.makedirs(out, exist_ok=True)
    manifest = {"ricursive_full.safetensors": save(os.path.join(out, "ricursive_full.safetensors"), tensors)}
    manifest["stream"] = dict(
        dtype=args.dtype, device=args.device, tokens=ids.shape[1], logits_rows=min(args.logits_rows, ids.shape[1]),
        tap_rows=min(args.tap_rows, ids.shape[1]), taps=sorted(tap_layers), **stats,
    )
    identity = dict(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers, num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim, sliding_window=cfg.sliding_window,
    )
    local = [os.path.join(args.checkpoint_dir, shard) for shard in sorted(set(ckpt.shard_of.values()))[:1]] if isinstance(ckpt, StreamedCheckpoint) else []
    finish(out, manifest, identity, args, checkpoint=args.checkpoint_dir, files=local)


def full_check(args, cfg_mod, mdl_mod):
    """`streamed_forward` over a sharded tiny checkpoint equals the whole-model forward, reading each tensor once."""
    import tempfile

    cfg, model = tiny_model(args, cfg_mod, mdl_mod)
    ids = torch.randint(0, TINY["vocab_size"], (1, TINY_PROMPT_LEN))
    whole, _taps, _logits = run(model, ids)
    state = model.state_dict()
    with tempfile.TemporaryDirectory(dir=args.out if os.path.isdir(args.out) else None) as root:
        weight_map = {}
        for shard, prefix in enumerate(["model.layers.0.", "model.layers.1.", "model.layers.2.", ""]):
            names = [n for n in state if (n.startswith(prefix) if prefix else not n.startswith("model.layers."))]
            file = f"model-{shard:05d}.safetensors"
            save_file({n: state[n].detach().clone().contiguous() for n in names}, os.path.join(root, file))
            weight_map.update({n: file for n in names})
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)
        ckpt = StreamedCheckpoint(root)
        tap_layers = parse_taps("every", cfg.num_hidden_layers)
        got, stats = streamed_forward(ckpt, cfg, mdl_mod, ids, "cpu", torch.float32, tap_layers, TINY_PROMPT_LEN, TINY_PROMPT_LEN)
        # Reading the logits of the last 5 rows only keeps the same rows as the whole run's last 5.
        last5, _ = streamed_forward(StreamedCheckpoint(root), cfg, mdl_mod, ids, "cpu", torch.float32, set(), 5, 5)
        worst = 0.0
        for k in range(cfg.num_hidden_layers):
            want = whole[f"hidden.{k}"]
            worst = max(worst, (got[f"hidden.{k}"] - want).abs().max().item() / want.abs().max().item())
        for name in ("final_hidden", "logits"):
            want = whole[name]
            worst = max(worst, (got[name] - want).abs().max().item() / want.abs().max().item())
        tail = (last5["logits"] - whole["logits"][-5:]).abs().max().item() / whole["logits"].abs().max().item()
        reads = sorted(set(ckpt.reads.values()))
        print(f"full-check: worst scaled error {worst:.3e}, last-5-rows logits {tail:.3e}, reads per tensor {reads}, "
              f"resident layers {stats['max_resident_layers']}, {len(ckpt.reads)} of {len(ckpt.shard_of)} tensors read", flush=True)
        assert worst <= 1e-6 and tail <= 1e-6, "the layer-streamed forward differs from the whole-model forward"
        assert reads == [1], f"every tensor is read exactly once, got {reads}"
        assert stats["max_resident_layers"] == 1
        assert len(ckpt.reads) == len(ckpt.shard_of), "a tensor of the checkpoint was never read"
        # The check can fail: a stream that skips the embedding scale is far from the whole-model forward.
        cfg.embed_scale = None
        wrong, _ = streamed_forward(StreamedCheckpoint(root), cfg, mdl_mod, ids, "cpu", torch.float32, set(), 1, TINY_PROMPT_LEN)
        effect = (wrong["logits"] - whole["logits"]).abs().max().item() / whole["logits"].abs().max().item()
        print(f"full-check: without the embedding scale the logits move by {effect:.3e}", flush=True)
        assert effect > 1e-3, "the streamed forward does not depend on the embedding scale: the check is blind"

        # The same checkpoint over HTTP range requests (a local server with range support): the same stream, nothing on disk.
        cfg.embed_scale = TINY["embed_scale"]
        import functools
        import http.server
        import threading

        class Ranged(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                span = self.headers.get("Range")
                path = self.translate_path(self.path)
                data = open(path, "rb").read()
                if span:
                    a, b = (int(v) for v in span.removeprefix("bytes=").split("-"))
                    data = data[a : b + 1]
                    self.send_response(206)
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Ranged, directory=root))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            remote = HttpCheckpoint(f"http://127.0.0.1:{server.server_port}/")
            over_http, _ = streamed_forward(remote, cfg, mdl_mod, ids, "cpu", torch.float32, set(), 4, TINY_PROMPT_LEN)
        finally:
            server.shutdown()
        gap = (over_http["logits"] - got["logits"]).abs().max().item() / got["logits"].abs().max().item()
        print(f"full-check: over HTTP range requests the logits differ from the directory's by {gap:.3e}; {len(remote.reads)} tensors fetched", flush=True)
        assert gap == 0.0 and sorted(set(remote.reads.values())) == [1]

        # The command path itself: the same directory (with its config) through `full`, then the dump read back.
        with open(os.path.join(root, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f)
        with open(os.path.join(root, "ids.json"), "w") as f:
            json.dump(ids[0].tolist(), f)
        dump = argparse.Namespace(**{**vars(args), "checkpoint_dir": root, "ids_file": os.path.join(root, "ids.json"), "prompt_file": None,
                                     "max_tokens": 4096, "device": "cpu", "dtype": "float32", "taps": "0,3", "tap_rows": 4, "logits_rows": 6, "out": os.path.join(root, "out")})
        full(dump, cfg_mod, mdl_mod)
        from safetensors.torch import load_file

        written = load_file(os.path.join(root, "out", "full", "ricursive_full.safetensors"))
        assert sorted(written) == ["final_hidden", "hidden.0", "hidden.3", "input_ids", "logits"], sorted(written)
        assert written["logits"].shape == (6, TINY["vocab_size"]) and written["hidden.3"].shape == (4, TINY["hidden_size"])
        assert (written["logits"] - whole["logits"][-6:]).abs().max().item() <= 1e-6 * whole["logits"].abs().max().item()
        with open(os.path.join(root, "out", "full", "manifest.json")) as f:
            stream = json.load(f)["stream"]
        assert stream["tokens"] == TINY_PROMPT_LEN and stream["max_resident_layers"] == 1, stream
        print("full-check: the `full` command wrote its dump and manifest as specified", flush=True)


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
    ap.add_argument("mode", choices=["tiny", "real", "grads", "real-grads", "real-lora", "full", "full-check"])
    ap.add_argument("--modeling-dir", required=True, help="directory holding modeling_ricursive.py and configuration_ricursive.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", help="real: the partial safetensors holding embed, norm, lm_head and layers 0 and 4")
    ap.add_argument("--config", help="real: the released config.json")
    ap.add_argument("--tokenizer", help="real: the released tokenizer.json")
    ap.add_argument("--checkpoint-dir", help="full: the released checkpoint directory (config.json, model.safetensors.index.json and its shards), or the URL of a repository'\''s files (https://huggingface.co/<repo>/resolve/main/), read by range requests without touching the disk")
    ap.add_argument("--prompt-file", help="full: a UTF-8 text file, tokenized with --tokenizer")
    ap.add_argument("--ids-file", help="full: a JSON list of token ids (instead of --prompt-file)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="full: keep the first N prompt tokens (default 4096)")
    ap.add_argument("--device", default="cpu", help="full: where the layers run (cpu, cuda:0)")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="full: the reference's compute dtype (bfloat16: the release as served; float32: weights promoted)")
    ap.add_argument("--taps", default="every", help="full: layers whose input stream is dumped: every, none, or a comma list (80 is the stream entering the final norm)")
    ap.add_argument("--tap-rows", type=int, default=16, help="full: the last N positions of each tapped stream (default 16)")
    ap.add_argument("--logits-rows", type=int, default=64, help="full: the last N positions of the logits (default 64; 100352 floats a row)")
    ap.add_argument("--seed", type=int, default=None, help="tiny: weight and prompt seed (default: random, printed and recorded in the manifest)")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(31)
    print(f"seed {args.seed}", flush=True)
    cfg_mod, mdl_mod = load_reference(os.path.abspath(args.modeling_dir))
    if args.mode in ("real", "real-grads", "real-lora") and not (args.checkpoint and args.config and args.tokenizer):
        ap.error("the real modes need --checkpoint, --config and --tokenizer")
    {"tiny": tiny, "real": real, "grads": grads, "real-grads": real_grads, "real-lora": real_lora, "full": full, "full-check": full_check}[args.mode](args, cfg_mod, mdl_mod)


if __name__ == "__main__":
    main()
