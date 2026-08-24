#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 duration-head reference goldens.

Runs the OFFICIAL `ltx_core.duration_head.duration_head.DurationHead` (CPU,
fp32) with the REAL `ltx-2.5-duration-head-bf16.safetensors` weights (15
tensors, ~4MB) on deterministic synthetic "pooled connector output" token
sequences, dumping:

  duration_head.safetensors   video_tokens, audio_tokens -> per-modality
                               projected+tagged tokens -> pooled attention
                               output -> MLP hidden -> log-duration -> duration
  manifest.json                shapes, sha256, run params, versions

This module never runs the real 22B DiT / Gemma-4 text encoder that would
normally produce `video_tokens`/`audio_tokens` (the embeddings-Connector
output - see the reference module's own docstring) - out of scope for this
milestone (M8 part 1 only covers the upscalers + this head's OWN math). The
synthetic inputs below are deterministic random tensors at the real
`(cross_attention_dim)` widths the checkpoint's `config.transformer` records
(4096 video / 2048 audio), a handful of tokens each - exactly what the task
this dumper backs asked for: proving this module's own math, not an
end-to-end pipeline.

## Self-validation inside the dumper (two independent code paths, not a
## repeat of the same call)

`AttentionPooler.forward` delegates to `torch.nn.MultiheadAttention`, an
opaque fused op. This dumper ALSO computes the identical pooled output via an
explicit from-scratch reimplementation (`manual_mha`, below) that unpacks
`in_proj_weight`/`in_proj_bias` into Q/K/V, splits heads, computes scaled-
dot-product attention (`scale = 1/sqrt(head_dim)`, softmax over the KEY axis)
and applies `out_proj` by hand - the exact sequence of ops
`crates/ltxv/src/duration_head.rs`'s host-math port needs to replicate. The
two are asserted to agree tightly (`cosine >= 1 - 1e-6`, `nn.MultiheadAttention`
is not required to be BIT-identical to a hand decomposition of the same math),
which pins the head-split / scale / softmax-axis convention empirically
rather than by reading PyTorch's docstring alone.

Fresh-module determinism (build+load a second time from scratch, repeat the
whole forward, assert bit-identical) is the second self-validation, same
convention as every other `ltxv_*_dump_reference.py` in this repo.

Usage:
  python tools/goldens/ltxv_duration_head_dump_reference.py \\
      --weights /path/to/ltx-2.5-duration-head-bf16.safetensors \\
      --out testdata/golden/ltxv/duration_head [--seed 42]
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

# `LTXV_REFERENCE_ROOT` overrides for a checkout elsewhere; the default is
# repo-relative (`scratchpad/reference/ltxv/`, gitignored), never a
# machine-specific absolute path.
_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "scratchpad" / "reference" / "ltxv")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

from ltx_core.duration_head.duration_head import DurationHead  # noqa: E402
from ltx_core.duration_head.model_configurator import DurationHeadConfigurator  # noqa: E402
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402


def det_tokens(t, dim, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, t, dim, generator=g)


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                      "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {name}: " + ", ".join(f"{k}{list(v.shape)}"
                                        for k, v in sorted(tensors.items())), flush=True)


def agree(name, a, b, tol):
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    cos = F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()
    print(f"  self-validate {name}: max abs {d:.3e} / scale {scale:.3g} = {rel:.2e}, "
          f"cosine {cos:.9f} (tol {tol:g})", flush=True)
    assert rel <= tol, f"{name}: disagree by {rel:.3e} relative"


def manual_mha(queries, tokens, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, num_heads):
    """From-scratch `nn.MultiheadAttention(queries, tokens, tokens)` reimplementation:
    unpack the packed QKV projection, split heads, scaled-dot-product attention,
    `out_proj`. See this module's docstring for why this exists (it is the spec
    the Rust host-math port replicates, not incidental test coverage)."""
    b, tq, d = queries.shape
    tk = tokens.shape[1]
    wq, wk, wv = in_proj_weight.chunk(3, dim=0)
    bq, bk, bv = in_proj_bias.chunk(3, dim=0)
    q = queries @ wq.T + bq
    k = tokens @ wk.T + bk
    v = tokens @ wv.T + bv
    hd = d // num_heads
    q = q.view(b, tq, num_heads, hd).transpose(1, 2)
    k = k.view(b, tk, num_heads, hd).transpose(1, 2)
    v = v.view(b, tk, num_heads, hd).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    probs = scores.softmax(dim=-1)
    ctx = probs @ v
    ctx = ctx.transpose(1, 2).reshape(b, tq, d)
    return ctx @ out_proj_weight.T + out_proj_bias


def build_model(weights_path):
    loader = SafetensorsModelStateDictLoader()
    metadata = loader.metadata(weights_path)
    model = DurationHeadConfigurator.from_metadata(metadata)
    sd = loader.load(weights_path, None)
    # Checkpoint tensors carry a `duration_head.` prefix (see
    # `DURATION_HEAD_KEY_OPS` in `duration_head/model_configurator.py`); strip
    # it by hand here rather than importing the SDOps constant, since the
    # loader's `load(path, None)` above already returned bare-of-any-rename
    # values and only the prefix needs stripping.
    sd_stripped = {k[len("duration_head."):]: v for k, v in sd.sd.items() if k.startswith("duration_head.")}
    model.load_state_dict({k: v.to(torch.float32) for k, v in sd_stripped.items()}, strict=True)
    model.eval().requires_grad_(False)
    return model, metadata.get("config", {})


def run(model, video_tokens, audio_tokens):
    duration = model(video_tokens=video_tokens, audio_tokens=audio_tokens)

    video_proj = model.video_input_proj(video_tokens) + model.video_modality_emb
    audio_proj = model.audio_input_proj(audio_tokens) + model.audio_modality_emb
    tokens = torch.cat([video_proj, audio_proj], dim=1)

    pooler = model.attention_pooler
    queries = pooler.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
    pooled_ref, _ = pooler.cross_attn(queries, tokens, tokens, need_weights=False)
    pooled_manual = manual_mha(
        queries, tokens,
        pooler.cross_attn.in_proj_weight, pooler.cross_attn.in_proj_bias,
        pooler.cross_attn.out_proj.weight, pooler.cross_attn.out_proj.bias,
        pooler.cross_attn.num_heads,
    )

    pooled_flat = pooled_ref.reshape(pooled_ref.shape[0], -1)
    hidden = F.gelu(model.mlp_hidden(pooled_flat), approximate="tanh")
    log_duration = model.mlp_out(hidden).squeeze(-1)

    return {
        "duration": duration, "video_proj": video_proj, "audio_proj": audio_proj,
        "tokens": tokens, "pooled": pooled_ref, "pooled_manual": pooled_manual,
        "hidden": hidden, "log_duration": log_duration,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="ltx-2.5-duration-head-bf16.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-tokens", type=int, default=4)
    ap.add_argument("--audio-tokens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    model, cfg = build_model(args.weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"built duration head ({n_params} params): {cfg}", flush=True)

    video_tokens = det_tokens(args.video_tokens, model.video_input_proj.in_features, args.seed)
    audio_tokens = det_tokens(args.audio_tokens, model.audio_input_proj.in_features, args.seed + 1)

    r1 = run(model, video_tokens, audio_tokens)

    # ---- self-validation 1: manual MHA decomposition vs nn.MultiheadAttention
    agree("pooled (manual MHA vs nn.MultiheadAttention)", r1["pooled_manual"], r1["pooled"], tol=1e-6)

    # ---- self-validation 2: fresh module instantiation, bit-identical ------
    model2, _ = build_model(args.weights)
    r2 = run(model2, video_tokens, audio_tokens)
    for k in r1:
        agree(f"fresh-instantiation {k}", r2[k], r1[k], tol=0.0)
    del model2, r2

    print(f"  duration: {r1['duration'].item():.6f} seconds", flush=True)

    tensors = {
        "video_tokens": video_tokens[0], "audio_tokens": audio_tokens[0],
        "video_proj": r1["video_proj"][0], "audio_proj": r1["audio_proj"][0],
        "tokens": r1["tokens"][0], "pooled": r1["pooled"][0],
        "hidden": r1["hidden"][0], "log_duration": r1["log_duration"],
        "duration": r1["duration"],
    }
    manifest = {
        "run": {"seed": args.seed, "video_tokens": args.video_tokens, "audio_tokens": args.audio_tokens,
                "weights": os.path.abspath(args.weights),
                "video_dim": model.video_input_proj.in_features, "audio_dim": model.audio_input_proj.in_features,
                "pooler_hidden_dim": model.pooler_hidden_dim,
                "num_heads": model.attention_pooler.cross_attn.num_heads,
                "duration_seconds": r1["duration"].item()},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    save(args.out, "duration_head.safetensors", tensors, manifest)

    # The two input projections' widths and the pooler geometry are what fix
    # every dumped tensor here; `num_heads` is in because the pooler's
    # cross-attention reshapes by it, so a checkpoint with the same dims but a
    # different head count produces same-shaped tensors with different values -
    # the failure that reads as a parity bug rather than a wrong pairing.
    manifest["source"] = source_block(
        checkpoint="Lightricks/LTX-2.5",
        files=[args.weights],
        hash_files=False,
        identity={
            "video_dim": int(model.video_input_proj.in_features),
            "audio_dim": int(model.audio_input_proj.in_features),
            "pooler_hidden_dim": int(model.pooler_hidden_dim),
            "num_heads": int(model.attention_pooler.cross_attn.num_heads),
        },
    )

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
