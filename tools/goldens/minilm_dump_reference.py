#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake staged all-MiniLM-L6-v2 goldens for `crates/decide`'s parity ladder.

Loads the released checkpoint through the real `transformers` `BertModel`, runs
a FIXED batch (CPU, fp32, eager attention) and saves one tensor per RUNG of the
ladder, so a parity failure names the stage that broke rather than reporting a
wrong number at the end:

  emb            post-embedding residual (word + position + token_type, LayerNorm)
  l0.attn_ctx    layer 0 self-attention context, BEFORE the output projection
  l0.attn_out    layer 0 after attention output + residual + LayerNorm
  l0.ffn_act     layer 0 after the intermediate projection and its GELU
  layer.{0..5}   every layer's output
  pooled_mean    mask-aware mean pooling, which is the sentence-transformer head

Token ids are fixed rather than tokenized, so forward parity is isolated from
tokenizer parity (`crates/data/tests/wordpiece_parity.rs` owns the other half).

The batch deliberately exercises, in one dump, the three things a hand-written
forward gets wrong independently of the arithmetic: BOTH token_type ids (row 0
is segment 0, row 1 is segment 1 - `crates/decide` separates its state and slot
roles through exactly this embedding), a padded row whose pad positions must be
masked out of attention AND out of the mean, and a batch dimension.

Swedish Embedded AB implements from-scratch transformer inference with
numerically proven parity against released checkpoints. If your team needs a
model reproduced on its own runtime rather than trusted, you can procure our
services by sending an email to info@swedishembedded.com.

usage: minilm_dump_reference.py [<hf_checkpoint_dir>] [<out_dir>]
"""
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import BertModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

DEFAULT_CKPT = os.path.expanduser(
    os.environ.get("BRAIN_MINILM_DIR", "~/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2"))
OUT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "decide", "tests", "golden")

# A fixed, arbitrary batch. Ids are < 30522; 101/102 are [CLS]/[SEP] so the rows
# look like real encodings, and 0 is [PAD].
IDS = [
    [101, 1045, 2572, 2145, 3403, 2006, 2026, 4003, 1029, 102, 7592, 2088, 1037, 2033, 4283, 102],
    [101, 2029, 2136, 2323, 5047, 2023, 102, 4003, 5508, 102, 0, 0, 0, 0, 0, 0],
]
MASK = [
    [1] * 16,
    [1] * 10 + [0] * 6,
]
# Row 0 is segment 0, row 1 is segment 1: both rows of `token_type_embeddings`
# are live, so a forward that ignores the tensor cannot match.
TYPES = [[0] * 16, [1] * 16]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    out = os.path.normpath(sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT)
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(0)
    model = BertModel.from_pretrained(ckpt, attn_implementation="eager", dtype=torch.float32)
    model.eval()

    ids = torch.tensor(IDS, dtype=torch.long)
    mask = torch.tensor(MASK, dtype=torch.long)
    types = torch.tensor(TYPES, dtype=torch.long)

    taps = {}
    l0 = model.encoder.layer[0]
    # The self-attention module returns the context BEFORE `attention.output`'s
    # projection; that split is the rung where a wrong head layout shows up
    # while the layer output can still look plausible.
    h = []
    h.append(l0.attention.self.register_forward_hook(
        lambda _m, _i, o: taps.__setitem__("l0.attn_ctx", (o[0] if isinstance(o, tuple) else o).detach().clone())))
    h.append(l0.attention.output.register_forward_hook(
        lambda _m, _i, o: taps.__setitem__("l0.attn_out", o.detach().clone())))
    h.append(l0.intermediate.register_forward_hook(
        lambda _m, _i, o: taps.__setitem__("l0.ffn_act", o.detach().clone())))

    with torch.no_grad():
        res = model(input_ids=ids, attention_mask=mask, token_type_ids=types, output_hidden_states=True)
    for x in h:
        x.remove()

    hs = res.hidden_states
    tensors = {"emb": hs[0].contiguous()}
    for i in range(len(hs) - 1):
        tensors[f"layer.{i}"] = hs[i + 1].contiguous()
    for k, v in taps.items():
        tensors[k] = v.contiguous()

    # sentence-transformers' pooling head: mean over UNMASKED positions only.
    m = mask.unsqueeze(-1).to(hs[-1].dtype)
    tensors["pooled_mean"] = ((hs[-1] * m).sum(1) / m.sum(1).clamp(min=1e-9)).contiguous()

    tensors["input_ids"] = ids.to(torch.int32)
    tensors["attention_mask"] = mask.to(torch.int32)
    tensors["token_type_ids"] = types.to(torch.int32)

    path = os.path.join(out, "minilm.safetensors")
    save_file(tensors, path)

    cfg = model.config
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "modules": "transformers.BertModel (real checkpoint, live run, eager attention)",
            "torch": torch.__version__,
            "source": source_block(
                checkpoint="sentence-transformers/all-MiniLM-L6-v2",
                files=[os.path.join(ckpt, "model.safetensors")],
                identity={
                    "hidden_size": cfg.hidden_size,
                    "num_hidden_layers": cfg.num_hidden_layers,
                    "num_attention_heads": cfg.num_attention_heads,
                    "intermediate_size": cfg.intermediate_size,
                    "vocab_size": cfg.vocab_size,
                    "max_position_embeddings": cfg.max_position_embeddings,
                    "type_vocab_size": cfg.type_vocab_size,
                },
            ),
            "files": {"minilm.safetensors": {
                "sha256": sha,
                "tensors": {k: list(v.shape) for k, v in tensors.items()},
            }},
        }, f, indent=2)
        f.write("\n")
    print(f"wrote {len(tensors)} tensors to {path}")
    for k in sorted(tensors):
        print(f"  {k:16s} {list(tensors[k].shape)}")


if __name__ == "__main__":
    main()
