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
    ap.add_argument("mode", choices=["tiny", "real", "grads", "real-grads", "real-lora"])
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
    if args.mode in ("real", "real-grads", "real-lora") and not (args.checkpoint and args.config and args.tokenizer):
        ap.error("the real modes need --checkpoint, --config and --tokenizer")
    {"tiny": tiny, "real": real, "grads": grads, "real-grads": real_grads, "real-lora": real_lora}[args.mode](args, cfg_mod, mdl_mod)


if __name__ == "__main__":
    main()
