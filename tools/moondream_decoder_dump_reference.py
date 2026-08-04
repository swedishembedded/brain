#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Partial-depth parity reference for the Moondream 3 text decoder.

The full 24-layer 9 B MoE model far exceeds RAM, so we validate the first 4 *real*
DENSE blocks (layers 0-3, before the MoE start_layer) end-to-end: the bespoke
parallel attn+MLP block with per-head tau temperature + partial RoPE that has no
standard-HF analogue. brain builds the identical 4-layer decoder and must match.

The moondream modeling code ships as loose .py files with relative imports and a
hyphenated dir name; we copy just the .py into an importable package. Only the four
blocks + wte + post_ln + lm_head are streamed from the checkpoint.

Outputs (little-endian):
  parity/moondream_dec_ref.bin     [T, vocab] logits (all positions)
  parity/moondream_dec_tokens.bin  [T] i32 token ids
"""
import glob
import os
import shutil
import sys
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import torch
from safetensors import safe_open

SRC = os.environ.get("BRAIN_MOONDREAM_CKPT", "/data/workspace/resources/vl/moondream3/hf/moondream3-preview")
OUT = os.environ.get("BRAIN_VL_PARITY_OUT", "/data/workspace/resources/vl/parity")
os.makedirs(OUT, exist_ok=True)
N = 4  # dense blocks (0-3, before MoE start_layer=4)

# Make the loose .py importable as package `md`.
pkgroot = tempfile.mkdtemp()
pkg = os.path.join(pkgroot, "md")
os.makedirs(pkg)
for py in glob.glob(f"{SRC}/*.py"):
    shutil.copy(py, pkg)
open(os.path.join(pkg, "__init__.py"), "w").close()
sys.path.insert(0, pkgroot)

from dataclasses import replace  # noqa: E402

from md.config import TextConfig  # noqa: E402
from md.text import build_text_model, text_decoder, text_encoder  # noqa: E402

# layers 0-3 are dense regardless; force the dense path with moe=None.
cfg = replace(TextConfig(), n_layers=N, moe=None)
w = build_text_model(cfg, torch.float32)

# Stream just the tensors we need (model.text.{blocks 0..N, wte, post_ln, lm_head}).
sd = {}
wte = None
with safe_open(f"{SRC}/model.safetensors" if os.path.exists(f"{SRC}/model.safetensors") else sorted(glob.glob(f"{SRC}/*.safetensors"))[0], "pt") as _probe:
    pass
for shard in sorted(glob.glob(f"{SRC}/*.safetensors")):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            if not k.startswith("model.text."):
                continue
            s = k[len("model.text."):]
            if s == "wte":
                wte = f.get_tensor(k).float()
            elif s.startswith("blocks."):
                if int(s.split(".")[1]) < N:
                    sd[s] = f.get_tensor(k).float()
            elif s.startswith("post_ln.") or s.startswith("lm_head."):
                sd[s] = f.get_tensor(k).float()

missing, unexpected = w.load_state_dict(sd, strict=False)
missing = [m for m in missing if not m.endswith("freqs_cis") and m != "wte"]
assert not missing, f"missing: {missing[:8]}"
assert wte is not None
w.wte = torch.nn.Parameter(wte)
for block in w.blocks:
    block.kv_cache = None  # no KV cache in the parity forward
w.eval()

tokens = [1, 9707, 11, 1879, 0, 1246, 525, 498]  # bos-ish + fixed ids
ids = torch.tensor([tokens])
T = len(tokens)
attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))[None, None]  # causal (prefix=1)
position_ids = torch.arange(T)
with torch.no_grad():
    x = text_encoder(ids, w)
    hidden = text_decoder(x, w, attn_mask, position_ids, cfg)  # [1, T, dim]
    # all-position logits: post_ln + lm_head per position.
    from md.layers import layer_norm

    normed = layer_norm(hidden[0], w.post_ln)  # [T, dim]
    logits = normed @ w.lm_head.weight.T + w.lm_head.bias  # [T, vocab]

logits.numpy().astype("<f4").tofile(f"{OUT}/moondream_dec_ref.bin")
np.array(tokens, dtype="<i4").tofile(f"{OUT}/moondream_dec_tokens.bin")
print(f"Moondream {N}-layer dense partial logits {tuple(logits.shape)} → {OUT}/moondream_dec_ref.bin")
