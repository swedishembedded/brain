#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a handful of `embed_tokens`/`lm_head` rows from the real
`Qwen/Qwen3.8-27B-FP8` checkpoint's `outside.safetensors` (M10's embedding/
lm_head spot-check milestone bullet).

Both tensors are plain BF16 (never FP8-quantized), `[vocab, hidden]`, ~2.5 GB
each - this dumper never decodes the WHOLE table, only the specific rows a
Rust test asks for, via `safetensors`' own lazy slice accessor (`get_slice`),
so peak host stays at a handful of `[hidden]` vectors.

Outputs (under `--out`, default `testdata/golden/qwen35/embed_lm_head/`):
  rows.safetensors   token_ids, embed_rows [N,hidden], lm_head_rows [N,hidden]
  manifest.json       sha256, run params, library versions

Usage:
  python tools/goldens/qwen35_dump_embed_lm_head_rows.py \
      --dir /path/to/Qwen3.8-27B-FP8 [--out DIR] [--ids 0,1,100,5000,128000,248319]
"""

import argparse
import hashlib
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def save(out_dir, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() if v.dtype.is_floating_point else v for k, v in tensors.items()}
    path = os.path.join(out_dir, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="real checkpoint directory (outside.safetensors)")
    ap.add_argument("--out", default=os.path.join("testdata", "golden", "qwen35", "embed_lm_head"))
    ap.add_argument("--ids", default="0,1,100,5000,128000,248319")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    token_ids = [int(x) for x in args.ids.split(",")]
    shard = os.path.join(args.dir, "outside.safetensors")
    with safe_open(shard, framework="pt") as f:
        embed = f.get_slice("model.language_model.embed_tokens.weight")
        head = f.get_slice("lm_head.weight")
        embed_rows = torch.stack([embed[i : i + 1, :][0].to(torch.float32) for i in token_ids])
        head_rows = torch.stack([head[i : i + 1, :][0].to(torch.float32) for i in token_ids])

    tensors = {"token_ids": torch.tensor(token_ids, dtype=torch.int32), "embed_rows": embed_rows, "lm_head_rows": head_rows}
    save(args.out, "rows.safetensors", tensors, manifest := {})
    manifest["_meta"] = {"token_ids": token_ids, "torch_version": torch.__version__, "checkpoint_dir_basename": os.path.basename(os.path.normpath(args.dir))}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote rows.safetensors + manifest.json -> {args.out}")


if __name__ == "__main__":
    main()
