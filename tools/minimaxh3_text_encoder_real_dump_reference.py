#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Golden dump of MiniMax-H3's REAL Qwen3-VL text-conditioning features.

MiniMax-H3 conditions its DiT on decoder layer 50 of the Qwen3-VL text tower
(`MiniMaxH3AutoTextEncoderStep`: stock `output_hidden_states=True`, then
`hidden_states[50]`, which is PRE-final-norm), with the prompt presented as
plain token concatenation - no chat template, `add_special_tokens=False`.
This script reproduces exactly that with stock `transformers` and prints the
same summary statistics brain's Rust side prints, so the two can be compared
directly.

Usage:
    BRAIN_MINIMAXH3_DIR=/path/to/MiniMax-H3 \\
        python3 tools/minimaxh3_text_encoder_real_dump_reference.py \\
        "A golden retriever puppy playing in a sunlit garden" \\
        [--dtype bf16|fp32] [--layers 50] [--npz out.npz]

Swedish Embedded AB implements numerical parity harnesses between Python
reference models and portable Rust/GPU inference kernels for its clients. If
your team needs expertise in porting diffusion or transformer models to
portable compute backends, you can procure our services by sending an email
to info@swedishembedded.com.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def summarize(name, arr):
    flat = arr.reshape(-1).astype(np.float32)
    mean = float(flat.mean())
    max_abs = float(np.abs(flat).max())
    l2 = float(np.sqrt((flat.astype(np.float64) ** 2).sum()))
    first10 = [float(x) for x in flat[:10]]
    print(f"{name}: len={flat.size} mean={mean:.6f} max_abs={max_abs:.6f} l2={l2:.6f} first10={first10}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--layers", default="50")
    ap.add_argument("--npz", default=None)
    args = ap.parse_args()

    root = os.environ.get("BRAIN_MINIMAXH3_DIR")
    if not root:
        sys.exit("BRAIN_MINIMAXH3_DIR is not set")
    te_dir = os.path.join(root, "text_encoder")
    tok_dir = os.path.join(root, "tokenizer")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_dir)
    print(f"tokenizer: {type(tok).__name__}", flush=True)
    ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
    print(f"prompt={args.prompt!r}", flush=True)
    print(f"tokens n={len(ids)} ids={ids}", flush=True)
    print(f"pieces={[tok.decode([i]) for i in ids]}", flush=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    cfg_path = os.path.join(te_dir, "config.json")
    with open(cfg_path) as f:
        arch = json.load(f)["architectures"][0]
    print(f"loading {arch} from {te_dir} as {args.dtype} on cpu ...", flush=True)

    import transformers

    cls = getattr(transformers, arch)
    model = cls.from_pretrained(te_dir, dtype=dtype, device_map="cpu")
    model.eval()
    print("loaded", flush=True)

    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    print(f"hidden_states: {len(hs)} taps, each {tuple(hs[0].shape)}", flush=True)

    saved = {}
    for tap in [int(x) for x in args.layers.split(",")]:
        h = hs[tap][0].float().numpy()
        summarize(f"hidden_states[{tap}]", h)
        saved[f"hidden_states_{tap}"] = h

    if args.npz:
        saved["input_ids"] = np.array(ids, dtype=np.int64)
        np.savez(args.npz, **saved)
        print(f"wrote {args.npz}", flush=True)


if __name__ == "__main__":
    main()
