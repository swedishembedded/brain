#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Partial-depth parity reference for the Qwen3-VL-4B text decoder.

The full 36-layer decoder is ~14 GB as f32 weights plus a second device copy inside
brain (~32 GB) — over the RAM here. So we validate the first N *real* Qwen3 blocks
end-to-end (embed → N blocks → final norm → tied head → logits): a genuine Qwen3
block (QK-norm, head_dim 128, GQA 32/8, SwiGLU 9728, θ=5e6) on real weights, at a
memory footprint that fits. brain builds the identical N-layer model and must match.

Outputs (little-endian):
  parity/qwenvl_dec_ref.bin     [T, vocab] partial-depth logits
  parity/qwenvl_dec_tokens.bin  [T] i32 token ids
"""
import glob
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import torch
from safetensors import safe_open
from transformers import Qwen3Config, Qwen3ForCausalLM

CKPT = os.environ.get("BRAIN_QWENVL_CKPT") or sys.exit("set BRAIN_QWENVL_CKPT=<Qwen3-VL-4B-Instruct hf checkpoint dir> (no baked-in default: this path is machine-specific)")
OUT = os.environ.get("BRAIN_VL_PARITY_OUT") or sys.exit("set BRAIN_VL_PARITY_OUT=<parity output dir> (no baked-in default: this path is machine-specific)")
os.makedirs(OUT, exist_ok=True)
N = 4  # blocks to validate

# Gather only the decoder tensors we need (embed, norm, layers 0..N), streamed.
sd = {}
LM = "model.language_model."
for shard in sorted(glob.glob(f"{CKPT}/*.safetensors")):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            if not k.startswith(LM):
                continue
            s = k[len(LM):]
            if s in ("embed_tokens.weight", "norm.weight"):
                sd["model." + s] = f.get_tensor(k).float()
            elif s.startswith("layers.") and int(s.split(".")[1]) < N:
                sd["model." + s] = f.get_tensor(k).float()

cfg = Qwen3Config(
    hidden_size=2560, num_hidden_layers=N, num_attention_heads=32, num_key_value_heads=8,
    head_dim=128, intermediate_size=9728, vocab_size=151936, rope_theta=5000000.0,
    rms_norm_eps=1e-6, tie_word_embeddings=True, max_position_embeddings=262144,
)
model = Qwen3ForCausalLM(cfg)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not [m for m in missing if m != "lm_head.weight"], f"missing: {[m for m in missing if m != 'lm_head.weight'][:6]}"
model.eval().float()

tokens = [151643, 9707, 11, 1879, 0, 1246, 525, 498]
with torch.no_grad():
    logits = model(torch.tensor([tokens])).logits[0]
logits.numpy().astype("<f4").tofile(f"{OUT}/qwenvl_dec_ref.bin")
np.array(tokens, dtype="<i4").tofile(f"{OUT}/qwenvl_dec_tokens.bin")
print(f"Qwen3-VL {N}-layer partial logits {tuple(logits.shape)} → {OUT}/qwenvl_dec_ref.bin")
