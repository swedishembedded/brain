#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump T5 encoder reference goldens for brain's `crates/t5` parity ladder.

The model is the **T5-XXL encoder** FLUX.1 uses as its second text encoder
(`FLUX.1-*/text_encoder_2`, HF `T5EncoderModel`, 24 layers x 4096, 64 heads of
64, gated-GELU FFN with d_ff=10240, relative-position attention bias).

Files written under `--out`:

  t5xxl/encoder.safetensors  per-STAGE taps of a real `T5EncoderModel` forward,
                             captured with forward hooks: embedding output,
                             every block output, block-0 internals (both
                             RMSNorms, q/k/v/o, wi_0/wi_1/gated/wo), the shared
                             relative-position bias, the bucket-id table, and
                             `last_hidden_state`.
  t5xxl/tokenizer.safetensors  T5 sentencepiece ids for a fixed string set.
  tiny/ckpt/model.safetensors  a RANDOM 2-layer T5 encoder in HF layout, and
  tiny/golden.safetensors      the same per-stage taps for it. Its dims exist to
                             break the two coincidences T5-XXL hides: at XXL
                             `num_heads == d_kv == 64` and
                             `num_heads * d_kv == d_model == 4096`, so swapping
                             the head count for the head width in a kernel
                             Params list, or using `d_model` where the attention
                             inner width belongs, is INVISIBLE in the
                             real-weights gate. The tiny model uses
                             `num_heads=2, d_kv=64, d_model=64`
                             (`inner = 128 != d_model`). It needs no checkpoint,
                             so `--model` may be omitted to regenerate it alone.
  manifest.json              shapes + sha256 per file, the reference config, the
                             run parameters and the recorded semantic findings.

Everything is CPU + fp32 with fixed seeds and fixed prompts; every tensor is
stored as f32 (brain's safetensors reader is F32/F16/BF16-only — the int32 id
and bucket tables are exactly representable).

Two semantic questions are SETTLED HERE by measurement rather than by argument
(porting-playbook 6), and the answers land in the manifest:

  * **attention mask** — diffusers' `FluxPipeline._get_t5_prompt_embeds` calls
    the encoder with NO `attention_mask`, so pad positions are attended as ordinary
    keys. The dumper runs the encoder both ways and records the difference, so the
    Rust side implements the contract FLUX actually uses instead of guessing.
  * **bucket coverage** — the run asserts that the (query, key) bucket table at
    this sequence length hits every one of the 32 buckets, i.e. the parity test
    exercises the whole bucketing function including the log branch and the
    clamp to `num_buckets - 1`.

Usage:
  python3 tools/t5_dump_reference.py \
      --model     /path/to/FLUX.1-Kontext-dev/text_encoder_2 \
      --tokenizer /path/to/FLUX.1-Kontext-dev/tokenizer_2 \
      --out       testdata/t5 [--seq-len 128]
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch
from safetensors.torch import save_file

# Fixed prompts. Two of very different length so both content rows and a long
# right-pad run are exercised (T5 pads with id 0 and FLUX passes no mask, so pad
# rows are real inputs, not inert).
PROMPTS = [
    "a red fox sitting on a mossy rock in a misty forest, morning light",
    "a photo of a cat",
]
TOKENIZER_STRINGS = [
    "a red fox sitting on a mossy rock in a misty forest, morning light",
    "a photo of a cat",
    "",
    "Hello, World!  MiXeD Case   and   collapsed\twhitespace",
    "digits 0123456789 and symbols #@$%^&*()_+-=[]{}|;':\",./<>?",
    "café naïve 你好 \U0001f98a emoji",
    "a " * 300 + "truncated tail",
]
SEED = 0


def save(out_dir, rel, tensors, manifest, extra=None):
    """Save `tensors` as f32 safetensors under `out_dir/rel` and manifest it."""
    tensors = {
        k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()
    }
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    entry = {
        "sha256": sha,
        "bytes": os.path.getsize(path),
        "dtype": "F32",
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
    }
    if extra:
        entry.update(extra)
    manifest["files"][rel] = entry
    print(f"wrote {rel}: {len(tensors)} tensors, {os.path.getsize(path) / 1e6:.1f} MB",
          flush=True)


class Taps:
    """Collect module inputs/outputs by name; every tap is a forward hook."""

    def __init__(self):
        self.t = {}
        self.h = []

    def out(self, name, module, index=0):
        def hook(_m, _a, o):
            o = o[index] if isinstance(o, tuple) else o
            self.t[name] = o.detach().clone()

        self.h.append(module.register_forward_hook(hook))

    def inp(self, name, module, index=0):
        def hook(_m, a, _o):
            self.t[name] = a[index].detach().clone()

        self.h.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self.h:
            h.remove()
        self.h = []


def host_buckets(t, num_buckets, max_distance):
    """The bucket table recomputed from the published formula, in plain Python.

    This is deliberately NOT `T5Attention._relative_position_bucket`: it is the
    independent re-derivation the Rust host implementation is written against,
    and it is asserted equal to the reference's table below. If the two ever
    disagree the dumper fails instead of blessing a wrong table.
    """
    half = num_buckets // 2
    max_exact = half // 2
    out = torch.zeros((t, t), dtype=torch.long)
    for i in range(t):
        for j in range(t):
            rel = j - i  # memory_position - query_position
            b = half if rel > 0 else 0
            a = abs(rel)
            if a < max_exact:
                b += a
            else:
                big = max_exact + int(
                    math.log(a / max_exact) / math.log(max_distance / max_exact)
                    * (half - max_exact)
                )
                b += min(big, half - 1)
            out[i, j] = b
    return out


def dump_encoder(model_dir, tok_dir, out_dir, manifest, seq_len):
    from transformers import AutoTokenizer, T5EncoderModel

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_json = json.load(f)
    assert cfg_json["architectures"] == ["T5EncoderModel"], cfg_json["architectures"]

    te = T5EncoderModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(tok_dir)
    cfg = te.config
    n_layers = cfg.num_layers
    n_heads = cfg.num_heads
    d_kv = cfg.d_kv
    print(f"loaded: {sum(p.numel() for p in te.parameters()) / 1e9:.3f} B params",
          flush=True)

    # FLUX tokenizes with padding="max_length" to `max_sequence_length` and
    # truncation, exactly as below.
    ti = tok(PROMPTS, padding="max_length", max_length=seq_len, truncation=True,
             return_tensors="pt")
    ids = ti.input_ids
    mask = ti.attention_mask
    assert ids.shape == (len(PROMPTS), seq_len), ids.shape

    enc = te.encoder
    blocks = enc.block
    assert len(blocks) == n_layers
    b0 = blocks[0]
    sa0 = b0.layer[0].SelfAttention
    ff0 = b0.layer[1].DenseReluDense
    assert sa0.has_relative_attention_bias, "block 0 must own the relative bias"
    for li in range(1, n_layers):
        assert not blocks[li].layer[0].SelfAttention.has_relative_attention_bias, \
            f"block {li} unexpectedly owns a relative bias"

    taps = Taps()
    taps.out("embed", enc.embed_tokens)
    taps.out("b0_attn_norm", b0.layer[0].layer_norm)
    taps.out("b0_q", sa0.q)
    taps.out("b0_k", sa0.k)
    taps.out("b0_v", sa0.v)
    taps.out("b0_attn_out", sa0.o)
    taps.out("b0_position_bias", sa0, index=1)
    taps.out("b0_attn_res", b0.layer[0])
    taps.out("b0_ff_norm", b0.layer[1].layer_norm)
    taps.out("b0_wi0", ff0.wi_0)
    taps.out("b0_wi1", ff0.wi_1)
    taps.inp("b0_gated", ff0.wo)
    taps.out("b0_ff_out", ff0.wo)
    for li in range(n_layers):
        taps.out(f"block{li}_out", blocks[li])
    taps.inp("final_norm_in", enc.final_layer_norm)

    with torch.no_grad():
        out = te(ids, output_hidden_states=True)
    taps.remove()

    hs = out.hidden_states
    lhs = out.last_hidden_state
    assert len(hs) == n_layers + 1, f"hidden_states {len(hs)} != {n_layers + 1}"
    assert torch.equal(hs[0], taps.t["embed"]), "hs[0] is not the embedding output"
    # T5Stack appends the hidden state BEFORE each block and appends the
    # POST-final_layer_norm state at the end, so hs[k] is block[k-1]'s output
    # only for k < n_layers, and hs[-1] is already normalized. (This differs
    # from CLIP, where the whole tuple is pre-final-LN — proven here, not
    # assumed: the first version of this dumper asserted CLIP's convention and
    # failed on hs[24].)
    for li in range(n_layers - 1):
        assert torch.equal(hs[li + 1], taps.t[f"block{li}_out"]), \
            f"hs[{li + 1}] is not block[{li}] out"
    last_block = taps.t[f"block{n_layers - 1}_out"]
    assert torch.equal(taps.t["final_norm_in"], last_block), \
        "final norm input != last block output"
    with torch.no_grad():
        assert torch.equal(enc.final_layer_norm(last_block), lhs), \
            "last_hidden_state != final_layer_norm(block[-1] out)"
    assert torch.equal(hs[-1], lhs), "hs[-1] != last_hidden_state"

    # --- structural facts, asserted rather than assumed -------------------
    # (1) no bias anywhere in the encoder, (2) no attention scaling.
    for name, p in te.named_parameters():
        assert not name.endswith(".bias"), f"unexpected bias parameter {name}"
    # scores are a bare q.k^T: replay block 0's attention from the tapped q/k/v
    # with NO 1/sqrt(d_kv) factor and require the tapped attention input to `o`
    # to come back exactly.
    B, T = ids.shape
    q = taps.t["b0_q"].view(B, T, n_heads, d_kv).transpose(1, 2)
    k = taps.t["b0_k"].view(B, T, n_heads, d_kv).transpose(1, 2)
    v = taps.t["b0_v"].view(B, T, n_heads, d_kv).transpose(1, 2)
    pb = taps.t["b0_position_bias"]
    with torch.no_grad():
        scores = torch.matmul(q, k.transpose(3, 2)) + pb
        probs = torch.softmax(scores.float(), dim=-1)
        ctx = torch.matmul(probs, v).transpose(1, 2).reshape(B, T, -1)
        replay = sa0.o(ctx)
    d_unscaled = (replay - taps.t["b0_attn_out"]).abs().max().item()
    with torch.no_grad():
        scores_s = torch.matmul(q, k.transpose(3, 2)) / math.sqrt(d_kv) + pb
        ctx_s = torch.matmul(torch.softmax(scores_s.float(), -1), v)
        replay_s = sa0.o(ctx_s.transpose(1, 2).reshape(B, T, -1))
    d_scaled = (replay_s - taps.t["b0_attn_out"]).abs().max().item()
    print(f"attention replay: unscaled max|d| {d_unscaled:.3e}, "
          f"WITH 1/sqrt(d_kv) {d_scaled:.3e}", flush=True)
    assert d_unscaled == 0.0, "T5 attention is not the unscaled q.k^T we replayed"
    assert d_scaled > 0.0, "the 1/sqrt(d) variant is indistinguishable — bad probe"

    # (3) the gated-GELU FFN is gelu_new(wi_0(x)) * wi_1(x).
    # `gelu_new` is HF's NewGELUActivation, i.e. the *explicit tanh form* that
    # brain's `gelu.wgsl` computes — NOT `F.gelu(approximate="tanh")`, whose
    # fused kernel differs in the last bits (measured 5.7e-7 here). Replay with
    # the explicit formula so this is an exact check of the right thing.
    x0 = taps.t["b0_wi0"]
    with torch.no_grad():
        act = 0.5 * x0 * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x0 + 0.044715 * torch.pow(x0, 3.0))))
        gated = act * taps.t["b0_wi1"]
    d_gate = (gated - taps.t["b0_gated"]).abs().max().item()
    print(f"gated-GELU replay max|d| {d_gate:.3e}", flush=True)
    assert d_gate == 0.0, "gated FFN is not gelu_new(wi_0)*wi_1"
    assert cfg.dense_act_fn == "gelu_new" and cfg.is_gated_act

    # (4) the relative-position bias is layer 0's, shared by every later layer,
    #     and equals embedding(bucket_table) permuted to (1, H, q, k).
    buckets_ref = sa0._relative_position_bucket(
        torch.arange(T)[None, :] - torch.arange(T)[:, None],
        bidirectional=True,
        num_buckets=cfg.relative_attention_num_buckets,
        max_distance=cfg.relative_attention_max_distance,
    )
    buckets = host_buckets(T, cfg.relative_attention_num_buckets,
                           cfg.relative_attention_max_distance)
    assert torch.equal(buckets, buckets_ref), \
        "independently derived bucket table != transformers'"
    # Bucket `num_buckets//2` (the "positive direction, distance 0" slot) is
    # STRUCTURALLY unreachable in the bidirectional encoder: rel > 0 already
    # implies |rel| >= 1. Everything else must be hit at this sequence length,
    # which is what makes the parity test exercise both signs, the exact branch,
    # the log branch and the clamp to num_buckets-1.
    nb = cfg.relative_attention_num_buckets
    seen = sorted(set(buckets.flatten().tolist()))
    reachable = [b for b in range(nb) if b != nb // 2]
    assert seen == reachable, f"seq_len {T} covers only buckets {seen}"
    with torch.no_grad():
        pb_ref = sa0.relative_attention_bias(buckets).permute(2, 0, 1).unsqueeze(0)
    assert torch.equal(pb_ref, pb), "position_bias != embedding(buckets) permuted"

    # (5) mask vs no-mask — the FLUX contract, measured.
    with torch.no_grad():
        out_masked = te(ids, attention_mask=mask).last_hidden_state
    diff = (out_masked - lhs).abs()
    content = mask.bool()
    d_content = diff[content].max().item()
    d_pad = diff[~content].max().item() if (~content).any() else 0.0
    print(f"mask vs no-mask: content rows max|d| {d_content:.3e}, "
          f"pad rows max|d| {d_pad:.3e}", flush=True)

    tensors = {
        "input_ids": ids.to(torch.int32),
        "attention_mask": mask.to(torch.int32),
        "relative_position_bucket": buckets.to(torch.int32),
        "embed": taps.t["embed"],
        "b0_attn_norm": taps.t["b0_attn_norm"],
        "b0_q": taps.t["b0_q"],
        "b0_k": taps.t["b0_k"],
        "b0_v": taps.t["b0_v"],
        "b0_position_bias": pb[0],
        "b0_attn_ctx": ctx,
        "b0_attn_out": taps.t["b0_attn_out"],
        "b0_attn_res": taps.t["b0_attn_res"],
        "b0_ff_norm": taps.t["b0_ff_norm"],
        "b0_wi0": taps.t["b0_wi0"],
        "b0_wi1": taps.t["b0_wi1"],
        "b0_gated": taps.t["b0_gated"],
        "b0_ff_out": taps.t["b0_ff_out"],
        "last_hidden_state": lhs,
        "last_hidden_state_masked": out_masked,
    }
    for li in range(n_layers):
        tensors[f"block{li}_out"] = taps.t[f"block{li}_out"]

    save(out_dir, "t5xxl/encoder.safetensors", tensors, manifest, extra={
        "reference": "transformers T5EncoderModel",
        "weights": os.path.abspath(model_dir),
        "config": {
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "d_kv": cfg.d_kv,
            "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads,
            "relative_attention_num_buckets": cfg.relative_attention_num_buckets,
            "relative_attention_max_distance": cfg.relative_attention_max_distance,
            "layer_norm_epsilon": cfg.layer_norm_epsilon,
            "feed_forward_proj": cfg.feed_forward_proj,
            "dense_act_fn": cfg.dense_act_fn,
            "is_gated_act": cfg.is_gated_act,
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "pad_token_id": cfg.pad_token_id,
            "eos_token_id": cfg.eos_token_id,
        },
        "run": {"batch": B, "seq_len": T, "prompts": PROMPTS},
        "findings": {
            "attention_scale": "NONE. Replaying block-0 attention as a bare "
                               f"q.k^T + bias reproduces sa.o's input exactly "
                               f"(max|d| {d_unscaled:.3e}); inserting the usual "
                               f"1/sqrt(d_kv) gives max|d| {d_scaled:.3e}",
            "position_bias": "computed once in block 0 "
                             "(has_relative_attention_bias) and passed to every "
                             "later block; == relative_attention_bias(bucket) "
                             "permuted (q,k,H) -> (1,H,q,k); NO RoPE and no "
                             "absolute position embedding anywhere",
            "bucket_coverage": f"seq_len {T} covers every REACHABLE bucket "
                               f"({len(reachable)} of {nb}: both signs, the "
                               "exact branch, the log branch and the clamp). "
                               f"Bucket {nb // 2} is structurally unreachable "
                               "in a bidirectional encoder (rel > 0 implies "
                               "|rel| >= 1)",
            "norm": "RMSNorm (T5LayerNorm): no mean subtraction, no bias, "
                    f"eps {cfg.layer_norm_epsilon}, applied in fp32; the "
                    "residual stream is NOT rescaled",
            "ffn": f"gated: {cfg.dense_act_fn}(wi_0(x)) * wi_1(x) -> wo; "
                   f"replay max|d| {d_gate:.3e}",
            "bias_params": "the encoder has NO bias parameter at all "
                           "(asserted over named_parameters)",
            "attention_mask": "diffusers' FluxPipeline._get_t5_prompt_embeds "
                              "passes NO attention_mask, so pad positions are "
                              "attended as ordinary keys. Measured effect of "
                              f"adding the mask: content rows max|d| "
                              f"{d_content:.3e}, pad rows max|d| {d_pad:.3e}. "
                              "`last_hidden_state` is the UNMASKED run (the "
                              "FLUX contract); `last_hidden_state_masked` is "
                              "the masked one, dumped for reference only",
        },
        "notes": {
            "hidden_states": "hs[0] = embedding output, hs[k] = block[k-1] "
                             "output for 0 < k < num_layers, and hs[-1] is "
                             "POST final_layer_norm (== last_hidden_state) — "
                             "NOT CLIP's all-pre-LN convention",
            "b0_attn_ctx": "attention context BEFORE sa.o, [B, T, H*d_kv] with "
                           "head-major channels (head h at [h*d_kv, (h+1)*d_kv))",
            "b0_gated": "gelu(wi_0(x)) * wi_1(x), i.e. the input of wo",
        },
    })
    return {"seq_len": T, "batch": B, "n_layers": n_layers}


def dump_tiny(out_dir, manifest):
    """A random 2-layer T5 encoder whose dims break T5-XXL's coincidences.

    XXL has `num_heads == d_kv == 64` and `num_heads * d_kv == d_model`, so the
    real-weights gate cannot tell a head-count/head-width swap — or a
    `d_model`-for-inner-width substitution — from a correct port. This fixture
    can: `num_heads=2, d_kv=64, d_model=64` gives `inner = 128 != d_model` and
    `heads != d_kv`. No pretrained checkpoint is involved; the weights are
    seeded random, and the norm gains are randomised too (the default init
    leaves every `layer_norm.weight` at exactly 1.0, which hides a dropped or
    mis-indexed gain).
    """
    from transformers import T5EncoderModel
    from transformers.models.t5.configuration_t5 import T5Config

    torch.manual_seed(SEED)
    cfg = T5Config(
        vocab_size=256, d_model=64, d_ff=128, d_kv=64, num_layers=2, num_heads=2,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
        layer_norm_epsilon=1e-6, feed_forward_proj="gated-gelu", dropout_rate=0.0,
        is_encoder_decoder=False, use_cache=False, tie_word_embeddings=False,
    )
    assert cfg.num_heads * cfg.d_kv != cfg.d_model and cfg.num_heads != cfg.d_kv
    te = T5EncoderModel(cfg).eval()
    with torch.no_grad():
        for name, p in te.named_parameters():
            is_norm = "layer_norm" in name
            p.copy_(torch.randn_like(p) * (0.5 if is_norm else 0.15) + (1.0 if is_norm else 0.0))

    # `state_dict()` carries `encoder.embed_tokens.weight` as a VIEW of
    # `shared.weight`; safetensors refuses to serialise the aliased pair, and a
    # `save_pretrained` checkpoint (the released text_encoder_2) ships only
    # `shared.weight` — so drop the alias and match the real layout.
    sd = {k: v.clone().contiguous() for k, v in te.state_dict().items()
          if k != "encoder.embed_tokens.weight"}
    save(out_dir, "tiny/ckpt/model.safetensors", sd, manifest, extra={
        "reference": "transformers T5EncoderModel (randomly initialised)",
        "why": "dims break XXL's heads==d_kv and heads*d_kv==d_model coincidences",
    })

    B, T = 2, 12
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    ids[1, 6:] = 0  # a right-pad run, as FLUX's padding produces
    enc = te.encoder
    b0, sa0 = enc.block[0], enc.block[0].layer[0].SelfAttention
    ff0 = b0.layer[1].DenseReluDense
    taps = Taps()
    taps.out("embed", enc.embed_tokens)
    taps.out("b0_attn_norm", b0.layer[0].layer_norm)
    taps.out("b0_q", sa0.q)
    taps.out("b0_k", sa0.k)
    taps.out("b0_v", sa0.v)
    taps.out("b0_attn_out", sa0.o)
    taps.out("b0_position_bias", sa0, index=1)
    taps.out("b0_attn_res", b0.layer[0])
    taps.out("b0_ff_norm", b0.layer[1].layer_norm)
    taps.out("b0_wi0", ff0.wi_0)
    taps.out("b0_wi1", ff0.wi_1)
    taps.inp("b0_gated", ff0.wo)
    taps.out("b0_ff_out", ff0.wo)
    for li in range(cfg.num_layers):
        taps.out(f"block{li}_out", enc.block[li])
    with torch.no_grad():
        out = te(ids)
    taps.remove()

    q = taps.t["b0_q"].view(B, T, cfg.num_heads, cfg.d_kv).transpose(1, 2)
    k = taps.t["b0_k"].view(B, T, cfg.num_heads, cfg.d_kv).transpose(1, 2)
    v = taps.t["b0_v"].view(B, T, cfg.num_heads, cfg.d_kv).transpose(1, 2)
    with torch.no_grad():
        probs = torch.softmax(
            (torch.matmul(q, k.transpose(3, 2)) + taps.t["b0_position_bias"]).float(), -1)
        ctx = torch.matmul(probs, v).transpose(1, 2).reshape(B, T, -1)
        # same unscaled-scores proof as the XXL dump, at dims where a d_kv/heads
        # swap would change the answer
        assert (sa0.o(ctx) - taps.t["b0_attn_out"]).abs().max().item() == 0.0

    tensors = {
        "input_ids": ids.to(torch.int32),
        "b0_position_bias": taps.t["b0_position_bias"][0],
        "b0_attn_ctx": ctx,
        "last_hidden_state": out.last_hidden_state,
    }
    for name in ["embed", "b0_attn_norm", "b0_q", "b0_k", "b0_v", "b0_attn_out",
                 "b0_attn_res", "b0_ff_norm", "b0_wi0", "b0_wi1", "b0_gated",
                 "b0_ff_out"]:
        tensors[name] = taps.t[name]
    for li in range(cfg.num_layers):
        tensors[f"block{li}_out"] = taps.t[f"block{li}_out"]
    save(out_dir, "tiny/golden.safetensors", tensors, manifest, extra={
        "config": {"vocab_size": cfg.vocab_size, "d_model": cfg.d_model,
                   "d_ff": cfg.d_ff, "d_kv": cfg.d_kv, "num_layers": cfg.num_layers,
                   "num_heads": cfg.num_heads},
        "run": {"batch": B, "seq_len": T, "seed": SEED},
    })


def dump_tokenizer(tok_dir, out_dir, manifest, seq_len):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_dir)
    enc = tok(TOKENIZER_STRINGS, padding="max_length", max_length=seq_len,
              truncation=True, return_tensors="pt")
    lens = [len(tok(s, truncation=True, max_length=seq_len).input_ids)
            for s in TOKENIZER_STRINGS]
    save(out_dir, "t5xxl/tokenizer.safetensors", {
        "ids_padded": enc.input_ids.to(torch.int32),
        "mask": enc.attention_mask.to(torch.int32),
        "len": torch.tensor(lens, dtype=torch.int32),
    }, manifest, extra={
        "reference": f"transformers {type(tok).__name__} (sentencepiece unigram)",
        "strings": TOKENIZER_STRINGS,
        "tokenizer": {
            "vocab_size": tok.vocab_size,
            "model_max_length": int(tok.model_max_length),
            "eos": tok.eos_token, "eos_id": int(tok.eos_token_id),
            "pad": tok.pad_token, "pad_id": int(tok.pad_token_id),
            "unk": tok.unk_token, "unk_id": int(tok.unk_token_id),
        },
        "notes": {
            "padding": f"padding='max_length', max_length={seq_len}, "
                       "truncation=True — the FLUX call shape",
            "eos": "T5 appends </s> (id 1); there is no BOS",
        },
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF T5EncoderModel dir "
                                    "(FLUX.1-*/text_encoder_2). Omit to write "
                                    "ONLY the tiny fixture, which needs no "
                                    "checkpoint.")
    ap.add_argument("--tokenizer", help="tokenizer dir (default: --model's sibling "
                                        "tokenizer_2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=128,
                    help="padded sequence length (128 covers all 32 buckets)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"files": {}, "params": {
        "prompts": PROMPTS, "seed": SEED, "seq_len": args.seq_len,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }}

    # The tiny fixture is checkpoint-free, so it goes first and always runs:
    # a broken environment fails here in a second instead of after loading
    # 19 GB of weights.
    dump_tiny(args.out, manifest)

    name = "manifest.json"
    if args.model:
        tok_dir = args.tokenizer or os.path.join(
            os.path.dirname(os.path.abspath(args.model)), "tokenizer_2")
        manifest["params"]["model"] = os.path.abspath(args.model)
        manifest["params"]["tokenizer"] = os.path.abspath(tok_dir)
        dump_tokenizer(tok_dir, args.out, manifest, args.seq_len)
        manifest["params"].update(dump_encoder(args.model, tok_dir, args.out,
                                               manifest, args.seq_len))
    else:
        # Never clobber a full manifest with a tiny-only run.
        name = "manifest-tiny.json"
        print("no --model: wrote the tiny fixture only", flush=True)

    with open(os.path.join(args.out, name), "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
