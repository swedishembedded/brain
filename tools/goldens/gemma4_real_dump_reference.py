#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""DIAGNOSTIC (not committed to the gate suite): real-weight, real-width,
reduced-depth Gemma-4 port correctness check. `gemma4_dump_reference.py`
proves the op sequence at TINY random dims with every structural FLAG set to
the real value; this proves the same op sequence at REAL width (hidden=3840,
head_dim=256/512, GQA/MQA, k_eq_v) on the FIRST 6 real layers (5 sliding + 1
full, the real 5:1 `sliding_window_pattern`'s minimal instance - the SAME
layer-type ratio the tiny dumper already uses, just with real weights this
time), loaded straight from the real 26 GB bf16 checkpoint. Never checked
against the actual reference at any scale beyond the tiny random-dim
config before this.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "resources" / "ltxv" / "source" / "packages" / "ltx-core" / "src"))

from transformers.models.gemma4_unified.modeling_gemma4_unified import (  # noqa: E402
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedTextModel,
)

LAYERS = 6  # 5 sliding + 1 full, the real 5:1 pattern's minimal instance
T = 12      # a plausible short real-prompt token count

REAL_CONFIG = dict(
    vocab_size=262_144,
    hidden_size=3840,
    intermediate_size=15360,
    num_hidden_layers=LAYERS,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=256,
    global_head_dim=512,
    num_global_key_value_heads=1,
    attention_k_eq_v=True,
    sliding_window=1024,
    hidden_activation="gelu_pytorch_tanh",
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_bias=False,
    attention_dropout=0.0,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
    max_position_embeddings=131072,
)


def real_tensor_names(num_layers):
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for l in range(num_layers):
        p = f"model.layers.{l}"
        names += [
            f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight",
            f"{p}.pre_feedforward_layernorm.weight", f"{p}.post_feedforward_layernorm.weight",
            f"{p}.layer_scalar",
            f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight", f"{p}.mlp.down_proj.weight",
            f"{p}.self_attn.q_proj.weight", f"{p}.self_attn.k_proj.weight", f"{p}.self_attn.o_proj.weight",
            f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.k_norm.weight",
        ]
        # The real header omits v_proj entirely on full-attention layers
        # (`attention_k_eq_v=True` - k_eq_v layers reuse the K projection as
        # V, so the checkpoint never stores a separate weight - matches
        # `Gemma4UnifiedTextModel`'s own `self_attn.v_proj = None` on those
        # layers). `layer_types[-1]` is always `full_attention` (config
        # `__post_init__`, checked above): layer 5 here.
        is_full = (l + 1) % 6 == 0
        if not is_full:
            names.append(f"{p}.self_attn.v_proj.weight")
    return names


def load_real_tensors(path, names):
    out = {}
    with safe_open(path, framework="pt") as f:
        for name in names:
            out[name] = f.get_tensor(name).to(torch.float32)
    return out


def build_model(sd_raw, seed):
    torch.manual_seed(seed)
    cfg = Gemma4UnifiedTextConfig(**REAL_CONFIG)
    model = Gemma4UnifiedTextModel(cfg)
    # Strip the `model.` prefix - `Gemma4UnifiedTextModel` is built directly
    # (not wrapped in a `...ForConditionalGeneration`), so its OWN
    # `state_dict()` keys have no `model.` prefix (confirmed against the
    # tiny dumper's own `model.state_dict()` dump, which uses bare
    # `layers.N....`/`embed_tokens.weight`/`norm.weight`).
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd_raw.items()}
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().requires_grad_(False)
    return model, cfg


class Taps:
    def __init__(self):
        self.acc, self.handles = {}, []

    def watch(self, name, module, pick=lambda o: o):
        def hook(_m, _i, o):
            self.acc[name] = pick(o).detach().clone()
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_with_taps(model, input_ids):
    taps = Taps()
    n = model.config.num_hidden_layers
    taps.watch("layer0.self_attn", model.layers[0].self_attn, pick=lambda o: o[0])
    taps.watch(f"layer{n - 1}.self_attn", model.layers[n - 1].self_attn, pick=lambda o: o[0])
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    taps.close()
    return out, dict(taps.acc)


def agree(label, a, b, tol=1e-6):
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    print(f"  self-validate {label}: max_abs {d:.3e} / scale {scale:.3g} = {rel:.2e} (tol {tol:g})", flush=True)
    assert rel <= tol, f"{label}: disagree by {rel:.3e} relative"


def save(out, name, tensors, manifest):
    from safetensors.torch import save_file
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    total = sum(v.numel() for v in tensors.values()) * 4 / 1e6
    print(f"wrote {name}: {len(tensors)} tensors, {total:.2f} MB", flush=True)
    manifest.setdefault("files", {})[name] = {k: list(v.shape) for k, v in tensors.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    names = real_tensor_names(LAYERS)
    print(f"loading {len(names)} real tensors from {args.weights} ...", flush=True)
    sd_raw = load_real_tensors(args.weights, names)
    total_mb = sum(v.numel() * 4 for v in sd_raw.values()) / 1e6
    print(f"  loaded {total_mb:.1f} MB (fp32)", flush=True)

    model, cfg = build_model(sd_raw, args.seed)
    print(f"real-width model built: hidden={cfg.hidden_size}, layers={LAYERS} (of the real checkpoint's 48), "
          f"layer_types={cfg.layer_types}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    input_ids = torch.randint(0, cfg.vocab_size, (1, T), generator=g)

    out, taps = run_with_taps(model, input_ids)
    hs = out.hidden_states
    assert len(hs) == LAYERS + 1
    assert not any(h.isnan().any() for h in hs), "NaN in hidden_states"
    assert torch.equal(hs[-1], out.last_hidden_state)

    # ---- self-validation: k_eq_v structural check -----------------------
    for i, lt in enumerate(cfg.layer_types):
        has_v = model.layers[i].self_attn.v_proj is not None
        if lt == "full_attention":
            assert not has_v, f"layer {i} (full_attention) must have NO v_proj"
        else:
            assert has_v, f"layer {i} (sliding_attention) must have its own v_proj"
    print("  self-validate k_eq_v: OK", flush=True)

    # ---- self-validation: fresh instantiation, bit-identical ------------
    model2, _ = build_model(sd_raw, args.seed)
    out2, _ = run_with_taps(model2, input_ids)
    agree("fresh-instantiation last_hidden_state", out2.last_hidden_state, out.last_hidden_state, tol=0.0)
    del model2

    position_ids = torch.arange(T).unsqueeze(0)
    sliding_cos, sliding_sin = model.rotary_emb(hs[0], position_ids, "sliding_attention")
    full_cos, full_sin = model.rotary_emb(hs[0], position_ids, "full_attention")
    for label, c, s in [("sliding", sliding_cos, sliding_sin), ("full", full_cos, full_sin)]:
        unit = c.double() ** 2 + s.double() ** 2
        max_dev = (unit - 1.0).abs().max().item()
        print(f"  self-validate RoPE({label}) cos^2+sin^2==1: max deviation {max_dev:.3e}", flush=True)
        assert max_dev < 1e-5

    tensors = {
        "input_ids": input_ids[0].to(torch.float32),
        "rope_sliding_cos": sliding_cos[0],
        "rope_sliding_sin": sliding_sin[0],
        "rope_full_cos": full_cos[0],
        "rope_full_sin": full_sin[0],
        "layer0_self_attn_out": taps["layer0.self_attn"][0],
        f"layer{LAYERS - 1}_self_attn_out": taps[f"layer{LAYERS - 1}.self_attn"][0],
        "last_hidden_state": out.last_hidden_state[0],
    }
    for k, h in enumerate(hs):
        tensors[f"hidden_states.{k}"] = h[0]

    manifest = {"run": {"seed": args.seed, "tokens": T, "layers": LAYERS,
                        "layer_types": cfg.layer_types, "real_config": REAL_CONFIG}}
    save(args.out, "gemma4_real_reduced.safetensors", tensors, manifest)
    with open(os.path.join(args.out, "manifest_real.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {args.out}/manifest_real.json", flush=True)


if __name__ == "__main__":
    main()
