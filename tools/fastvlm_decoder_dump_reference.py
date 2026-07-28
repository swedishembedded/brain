#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the FastVLM-0.5B Qwen2 *decoder* reference logits for a fixed token
sequence, so brain's Qwen decoder can be parity-checked on the real weights.

The FastVLM decoder is an unmodified Qwen2 (the Llava wrapper only adds the
vision tower + projector + image-token splice), so we load the `model.layers.*`
/ `model.embed_tokens` / `model.norm` / `lm_head` tensors straight into a vanilla
`Qwen2ForCausalLM` and run text-only. Outputs raw little-endian f32:
  parity/fastvlm_dec_ref.bin     [T, vocab] logits
  parity/fastvlm_dec_tokens.bin  [T] i32 token ids
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM

CKPT = "/data/workspace/resources/vl/fastvlm/hf/FastVLM-0.5B"
OUT = "/data/workspace/resources/vl/parity"
os.makedirs(OUT, exist_ok=True)

cfg = json.load(open(f"{CKPT}/config.json"))
qcfg = Qwen2Config(
    hidden_size=cfg["hidden_size"],
    num_hidden_layers=cfg["num_hidden_layers"],
    num_attention_heads=cfg["num_attention_heads"],
    num_key_value_heads=cfg["num_key_value_heads"],
    intermediate_size=cfg["intermediate_size"],
    vocab_size=cfg["vocab_size"],
    rope_theta=cfg["rope_theta"],
    rms_norm_eps=cfg["rms_norm_eps"],
    tie_word_embeddings=cfg["tie_word_embeddings"],
    max_position_embeddings=cfg["max_position_embeddings"],
    hidden_act=cfg["hidden_act"],
)
model = Qwen2ForCausalLM(qcfg)

sd = load_file(f"{CKPT}/model.safetensors")
dec = {
    k: v.float()
    for k, v in sd.items()
    if k.startswith("model.layers.") or k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
}
missing, unexpected = model.load_state_dict(dec, strict=False)
# Tied models may omit lm_head from the load (tied to embed_tokens) — that's fine.
missing = [m for m in missing if m != "lm_head.weight"]
assert not missing, f"missing decoder tensors: {missing[:8]}"
model.eval().float()

tokens = [151643, 9707, 11, 1879, 0, 1246, 525, 498]  # bos + arbitrary fixed ids
ids = torch.tensor([tokens], dtype=torch.long)
with torch.no_grad():
    logits = model(ids).logits[0]  # [T, vocab]

logits.numpy().astype("<f4").tofile(f"{OUT}/fastvlm_dec_ref.bin")
np.array(tokens, dtype="<i4").tofile(f"{OUT}/fastvlm_dec_tokens.bin")
print(f"dumped logits {tuple(logits.shape)} → {OUT}/fastvlm_dec_ref.bin")
