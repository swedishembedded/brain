#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump TimesFM-3 reference forwards for the brain parity ladder.

Two independent dumps, dumped before any Rust exists (goldens come first, so
the Rust side has a real target to gate against instead of a guess):

  A. A CHECKPOINT-FREE tiny config where every dimension differs from every
     other (heads != head_dim != model_dims, patch_in != patch_out !=
     n_quantiles, layers != heads) - random seeded weights, so a head-count/
     head-width swap or a sequence/variate-attention axis swap cannot pass by
     coincidence the way it could at the real 1280/16/80 shape (two of those
     three numbers share no factor at the real size either, but 1280 is
     divisible by both 16 and 80, so a transposed head split is not the kind
     of thing this dump alone would catch - the tiny config's dims are chosen
     pairwise coprime specifically to close that gap).
  B. A REAL-CHECKPOINT single-forward decode, multivariate: one target, one
     past-only covariate, one past-future covariate - exercising every `Role`
     brain's `Panel` will map onto this model.

Both dumps exercise sequence attention, variate attention, RoPE, QK-norm,
RevIN, iterative CPM refinement, linear detrending and stitching - every stage
a faithful Rust port has to reproduce.

Taps captured via forward hooks + one function wrap, never by
hand-reassembling intermediate math from the outputs alone - hooking the
module freezes the exact convention a hand assembly could otherwise get
subtly wrong:
  - pre_transformer_resblock: input, output
  - transformer_stack.layers[0, mid, last]: per-layer output
  - output_head: RAW logits (pre-revin-reverse, pre-CPM-refine substitution -
    forward()'s own aux dict only exposes the PRE-refine revin_stats, so the
    raw head output needs its own hook)
  - cpm_iterative_revin_refine: wrapped to also stash (refined_mu, refined_sigma)
  - decode()'s own return_aux_outputs=True dict (resblock_input/
    transformer_input/seq_attn_mask/transformer_output) and final horizon_logits

determinism: eval(), no_grad, single thread, and `use_sdpa=False` monkey-patched
onto every MultiHeadAttention instance so attention runs the manual masked-
softmax path (deterministic; PyTorch's SDPA kernel selection is not, across
backends - `porting.md` §1's "everything f32, fixed seeds" applies to the
attention implementation too, not only the inputs).

Outputs (into <out_dir>, normally `testdata/golden/timesfm3/` - gitignored,
regenerated on demand, never committed: it is large numeric data, not source):
  - manifest.json     shapes, per-tap RMS, the full numeric arrays this
                       ladder's rungs assert against, and golden_source.py's
                       `source` block (both parts).
  - tiny_*.npy         part A's full tensors, for ad-hoc inspection outside Rust.
  - real_*.npy         part B's full tensors, for ad-hoc inspection outside Rust.

Usage:
  BRAIN_TIMESFM3_REF=<google-research/timesfm checkout> \\
    python3 tools/goldens/timesfm3_dump_reference.py <checkpoint_dir> \\
    testdata/golden/timesfm3

<checkpoint_dir> is a local `google/timesfm-3.0-pytorch` checkout (what
`brain pull google/timesfm-3.0-pytorch` fetches) - not committed, not baked in
here: this path is machine-specific by nature, same as every other
`*_dump_reference.py` in this tree.

Not part of the build. Needs the reference checkout's own venv (torch,
huggingface_hub, safetensors).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REF = os.environ.get("BRAIN_TIMESFM3_REF") or sys.exit(
    "set BRAIN_TIMESFM3_REF=<google-research/timesfm checkout> (no baked-in default: this path is machine-specific)"
)
sys.path.insert(0, os.path.join(REF, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

import timesfm3  # noqa: E402
from timesfm3 import cpm_revin_refine as cpm_revin_refine_lib  # noqa: E402
from timesfm3 import transformer as transformer_lib  # noqa: E402

SEED = 1337


def force_deterministic_attention(model):
    """Every `MultiHeadAttention` submodule's `use_sdpa` is a plain instance
    attribute set at construction time (`transformer.py:177`), not part of the
    checkpoint - flipping it after `from_pretrained` forces the manual masked-
    softmax attention path, which (unlike SDPA's kernel selection) is
    reproducible across runs and across CPU/CUDA."""
    n = 0
    for m in model.modules():
        if isinstance(m, transformer_lib.MultiHeadAttention):
            m.use_sdpa = False
            n += 1
    return n


def sample(arr, k=64, seed=0):
    flat = np.asarray(arr).reshape(-1).astype(np.float64)
    if flat.size <= k:
        idx = np.arange(flat.size)
    else:
        idx = np.random.RandomState(seed).choice(flat.size, k, replace=False)
        idx.sort()
    return {"indices": idx.tolist(), "values": flat[idx].tolist()}


def tap(store, taps_manifest, name, t, full=False):
    """Record one tensor: full array into `store` (written as .npy, gitignored -
    for interactive local comparison against a much larger real forward) and
    shape/rms always into `taps_manifest` (committed, small, drives the CI
    parity gate). `full=True` additionally embeds the COMPLETE array in
    `taps_manifest` itself (still committed - only used for tensors small
    enough that this stays a small file, the boundary taps a checkpoint-free
    CI gate needs data for); otherwise a 64-value sample is embedded instead."""
    arr = t.detach().to(torch.float32).cpu().numpy()
    store[name] = arr
    entry = {
        "shape": list(arr.shape),
        "rms": float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))),
    }
    if full:
        entry["full"] = arr.reshape(-1).astype(np.float64).tolist()
    else:
        entry["sample"] = sample(arr, seed=hash(name) & 0xFFFFFFFF)
    taps_manifest[name] = entry


def wrap_cpm_refine(captured):
    """Wrap `cpm_iterative_revin_refine` to additionally stash its
    (refined_mu, refined_sigma) return - `forward()`'s own aux dict only
    exposes the PRE-refine running stats (the `revin_stats` local is never
    reassigned after the refine call), so the refined values are only
    observable by intercepting the call itself."""
    original = cpm_revin_refine_lib.cpm_iterative_revin_refine

    def wrapped(*args, **kwargs):
        mu, sigma = original(*args, **kwargs)
        captured["refined_mu"] = mu
        captured["refined_sigma"] = sigma
        return mu, sigma

    cpm_revin_refine_lib.cpm_iterative_revin_refine = wrapped
    return original


def run_case(model, prefix, store, taps_manifest, *, target, past_only, past_future, horizon, full=False):
    """Register hooks, run one `decode(..., return_aux_outputs=True)`, tap
    every stage, and return the final horizon logits. `full=True` embeds
    COMPLETE arrays for every tap in the committed manifest (only used for the
    checkpoint-free tiny config, small enough that this stays a small file);
    the real-checkpoint case embeds full arrays for only its two `core_forward`
    boundary taps (resblock_input, raw_logits) plus horizon_logits, all three
    small regardless of model size, and samples for the rest."""
    hooks = []
    layer_outs = {}

    def make_layer_hook(idx):
        def hook(_module, _inputs, output):
            layer_outs[idx] = output[0]  # (output_embeddings, caches, masks)

        return hook

    n_layers = len(model.transformer_stack.layers)
    watch_layers = sorted({0, n_layers // 2, n_layers - 1})
    for idx in watch_layers:
        hooks.append(model.transformer_stack.layers[idx].register_forward_hook(make_layer_hook(idx)))

    raw_logits_box = {}

    def head_hook(_module, _inputs, output):
        raw_logits_box["raw_logits"] = output

    hooks.append(model.output_head.register_forward_hook(head_hook))

    refined = {}
    original_refine = wrap_cpm_refine(refined)
    try:
        horizon_logits, forward_out = model.decode(
            target=target,
            past_only_covariates=past_only,
            past_future_covariates=past_future,
            horizon=horizon,
            return_aux_outputs=True,
        )
    finally:
        cpm_revin_refine_lib.cpm_iterative_revin_refine = original_refine
        for h in hooks:
            h.remove()

    tap(store, taps_manifest, f"{prefix}.resblock_input", forward_out["__call__:resblock_input"], full=True)
    tap(store, taps_manifest, f"{prefix}.transformer_input", forward_out["__call__:transformer_input"], full=full)
    tap(store, taps_manifest, f"{prefix}.transformer_output", forward_out["__call__:transformer_output"], full=full)
    for idx in watch_layers:
        tap(store, taps_manifest, f"{prefix}.layer{idx}_output", layer_outs[idx], full=full)
    tap(store, taps_manifest, f"{prefix}.raw_logits", raw_logits_box["raw_logits"], full=True)
    running_mean, running_std = forward_out["revin_stats"]
    tap(store, taps_manifest, f"{prefix}.revin_running_mean", running_mean, full=full)
    tap(store, taps_manifest, f"{prefix}.revin_running_std", running_std, full=full)
    if "refined_mu" in refined:
        tap(store, taps_manifest, f"{prefix}.cpm_refined_mu", refined["refined_mu"], full=full)
        tap(store, taps_manifest, f"{prefix}.cpm_refined_sigma", refined["refined_sigma"], full=full)
    tap(store, taps_manifest, f"{prefix}.horizon_logits", horizon_logits, full=True)
    return horizon_logits


def dump_tiny(store, manifest):
    """Part A: checkpoint-free, every dim pairwise distinct and pairwise
    coprime so a head-count/head-width or seq/variate-attention axis swap
    cannot pass by shape coincidence the way it could at the real 1280/16/80
    (all three share common factors there)."""
    # Every dimension pairwise distinct (and head_dim kept EVEN - RoPE's
    # split-half rotation divides it in two, so an odd head_dim is not a
    # tiny-config choice, it is a shape error): layers 3, heads 2, head_dim 6
    # -> model_dims 12, hidden_dims 14, patch_in 4, patch_out 8, quantiles 5,
    # max_variates 9. At the real 1280/16/80 shape several of these numbers
    # share factors (1280 = 16*80); here none of them do, so a head-count/
    # head-width swap or a sequence/variate axis swap cannot pass by shape
    # coincidence.
    torch.manual_seed(SEED)
    residual_cfg = timesfm3.ResidualBlockConfig(
        hidden_dims=12, output_dims=12, use_bias=False, activation="relu"
    )
    transformer_cfg = timesfm3.TransformerConfig(
        model_dims=12,
        hidden_dims=14,
        num_heads=2,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_bias=False,
        use_rope_seq=True,
        use_rope_var=False,
        ff_activation="relu",
        deterministic=True,
        causal_attention=True,
        use_memory_efficient_attention=False,
        max_variates=9,
        use_sdpa=False,
    )
    stacked_cfg = timesfm3.StackedTransformersConfig(num_layers=3, transformer=transformer_cfg)
    model = timesfm3.TimesFM3Torch(
        input_patch_len=4,
        output_patch_len=8,
        quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
        residual_block_config=residual_cfg,
        transformer_config=stacked_cfg,
        use_variate_attention=True,
        use_stitching=True,
        use_linear_detrending=True,
        use_iterative_cpm_revin=True,
    )
    model.eval()
    n_patched = force_deterministic_attention(model)
    assert n_patched == 2 * 3, f"expected 6 attention modules (seq+var x 3 layers), got {n_patched}"

    gen = torch.Generator().manual_seed(SEED)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=gen) * 0.05

    batch, context, horizon = 2, 16, 8  # 4 context patches of 4, 1 forecast patch of 8 (no horizon padding)
    target = torch.randn(batch, 2, context, generator=gen)
    past_only = torch.randn(batch, 1, context, generator=gen)
    past_future = torch.randn(batch, 1, context + horizon, generator=gen)

    weights = {name: p.detach().cpu().numpy() for name, p in model.state_dict().items()}
    for name, arr in weights.items():
        store_key = f"tiny.weight.{name}"
        store[store_key] = arr

    with torch.no_grad():
        run_case(model, "tiny", store, manifest, target=target, past_only=past_only, past_future=past_future, horizon=horizon, full=True)
    tap(store, manifest, "tiny.input.target", target, full=True)
    tap(store, manifest, "tiny.input.past_only", past_only, full=True)
    tap(store, manifest, "tiny.input.past_future", past_future, full=True)

    manifest["tiny_config"] = model.to_dict()
    manifest["tiny_weight_names"] = sorted(weights.keys())
    # Embedded fully (not just names) - the tiny model's total param count is
    # small enough that this is the one thing that lets a checkpoint-free CI
    # gate actually LOAD a model and run `core_forward`, rather than only
    # checking shapes against `tiny_weight_names`.
    manifest["tiny_weights"] = {name: arr.reshape(-1).astype(np.float64).tolist() for name, arr in weights.items()}
    return {
        "input_patch_len": 4, "output_patch_len": 8, "num_quantiles": 5,
        "num_layers": 3, "num_heads": 2, "model_dims": 12, "hidden_dims": 14,
        "context": context, "horizon": horizon, "batch": batch,
    }


def dump_real(checkpoint_dir, store, manifest):
    """Part B: the real 330M checkpoint, one multivariate decode covering all
    three `Role`s brain's `Panel` maps onto this model."""
    torch.manual_seed(SEED)
    model = timesfm3.TimesFM3Torch.from_pretrained(checkpoint_dir)
    model.eval()
    force_deterministic_attention(model)

    gen = torch.Generator().manual_seed(SEED)
    batch, context, horizon = 1, 192, 64  # 6 context patches of 32, 1 forecast patch (extract_len=64)
    target = torch.randn(batch, 1, context, generator=gen)
    past_only = torch.randn(batch, 1, context, generator=gen)
    past_future = torch.randn(batch, 1, context + horizon, generator=gen)

    with torch.no_grad():
        run_case(model, "real", store, manifest, target=target, past_only=past_only, past_future=past_future, horizon=horizon, full=False)
    tap(store, manifest, "real.input.target", target, full=True)
    tap(store, manifest, "real.input.past_only", past_only, full=True)
    tap(store, manifest, "real.input.past_future", past_future, full=True)
    manifest["real_config"] = model.to_dict()
    return {
        "input_patch_len": model.input_patch_len, "output_patch_len": model.output_patch_len,
        "num_quantiles": model.num_quantiles,
        "num_layers": model.transformer_config.num_layers,
        "num_heads": model.transformer_config.transformer.num_heads,
        "model_dims": model.transformer_config.transformer.model_dims,
        "hidden_dims": model.transformer_config.transformer.hidden_dims,
        "context": context, "horizon": horizon, "batch": batch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir", help="local google/timesfm-3.0-pytorch checkout (config.json + model.safetensors)")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.set_num_threads(1)
    store = {}
    manifest = {"versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}}

    tiny_identity = dump_tiny(store, manifest)
    real_identity = dump_real(args.checkpoint_dir, store, manifest)

    weights_file = os.path.join(args.checkpoint_dir, "model.safetensors")
    manifest["source"] = {
        "tiny": source_block(identity=tiny_identity, checkpoint="checkpoint-free (random seeded weights)"),
        "real": source_block(identity=real_identity, checkpoint="google/timesfm-3.0-pytorch", files=(weights_file,)),
    }

    for name, arr in store.items():
        np.save(os.path.join(args.out_dir, name.replace("/", "_") + ".npy"), arr)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {len(store)} tensors + manifest.json to {args.out_dir}")


if __name__ == "__main__":
    main()
