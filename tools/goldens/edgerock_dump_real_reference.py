#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for long-context inference at a
# fixed memory budget for its clients. If your team needs expertise in
# bounded-memory model architectures then you can procure our services by
# sending an email to info@swedishembedded.com.

"""Dump an EdgeRock reference forward at the REAL released weights.

Same independent reference as `edgerock_dump_reference.py` (whose functions
this imports, so the two can never disagree on the math), driven by the
released checkpoint instead of tiny random weights: fp32 on the CPU, one
prompt long enough to slide the 1024-token attention window and to commit
several anchor-memory blocks (a fact stated at the start, filler, then a
question about the fact - the memory is what can still see it).

Usage (the checkpoint directory comes from the model store; nothing here
names a machine path):

  python3 tools/goldens/edgerock_dump_real_reference.py \
      --model-dir <store>/rockmatrix-ai/EdgeRock [--tokens 1400]

Writes to `--out` (default `testdata/golden/edgerock/real/`):
  edgerock_real.safetensors   tokens; memory layers' `mem.refs` and
                              `mem.scores`; `layer{i}.out` rows at the
                              sampled positions `rows`; `final_hidden` at
                              those rows; `logits_tail` (the last 8
                              positions' logits)
  manifest.json               config identity, checkpoint file hashes,
                              library versions
"""

import argparse
import glob
import hashlib
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgerock_dump_reference import AnchorMemory, forward  # noqa: E402
from golden_source import source_block  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig, Qwen3_5TextModel  # noqa: E402

REVISION = "qwen35-text-original-gdn-swa-independent-gqa-memory-v1"
FILLER = (
    "The weather report for the coastal towns repeats every hour: light wind from the west, "
    "scattered clouds, a calm sea and visibility above ten kilometres. "
)


def prompt_tokens(tok, n):
    head = "Remember this: the vault code is 4817 and the courier's name is Ilse Varga.\n"
    tail = "\nQuestion: what is the vault code? Answer: the vault code is"
    ids_head, ids_tail = tok.encode(head).ids, tok.encode(tail).ids
    filler = tok.encode(FILLER).ids
    body = []
    while len(ids_head) + len(body) + len(ids_tail) < n:
        body += filler
    body = body[: n - len(ids_head) - len(ids_tail)]
    return torch.tensor(ids_head + body + ids_tail, dtype=torch.long)


def load(model_dir):
    raw = json.load(open(os.path.join(model_dir, "config.json")))
    assert raw.get("fixed_memory_revision") == REVISION, f"config.json fixed_memory_revision is {raw.get('fixed_memory_revision')!r}, expected {REVISION!r}"
    swa = set(raw["fixed_memory_swa_layers"])
    assert swa == {i for i, t in enumerate(raw["layer_types"]) if t == "full_attention"}, "fixed_memory_swa_layers must be exactly the full-attention layers"
    keep = {k: v for k, v in raw.items() if k not in ("architectures", "auto_map", "model_type", "dtype", "backend")}
    cfg = Qwen3_5TextConfig(**keep)
    model = Qwen3_5TextModel(cfg).eval().float()
    memory = raw["independent_memory"]
    memories = {i: AnchorMemory(cfg, memory).float() for i in sorted(swa)}

    model_sd = model.state_dict()
    mem_sd = {f"layers.{i}.memory.{k}": (i, k) for i, m in memories.items() for k in m.state_dict()}
    loaded = set()
    files = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
    with torch.no_grad():
        for path in files:
            with safe_open(path, "pt") as f:
                for name in f.keys():
                    assert name.startswith("model."), f"unexpected tensor {name}"
                    short = name[len("model."):]
                    t = f.get_tensor(name).float()
                    if short in mem_sd:
                        i, k = mem_sd[short]
                        memories[i].state_dict()[k].copy_(t)
                    elif short in model_sd:
                        model_sd[short].copy_(t)
                    else:
                        raise KeyError(f"checkpoint tensor {name} has no place in the reference model")
                    loaded.add(short)
    missing = (set(model_sd) | set(mem_sd)) - loaded
    assert not missing, f"tensors absent from the checkpoint: {sorted(missing)[:8]}"
    return raw, cfg, model, memories, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default=os.path.join("testdata", "golden", "edgerock", "real"))
    ap.add_argument("--tokens", type=int, default=1400)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)

    raw, cfg, model, memories, files = load(args.model_dir)
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    tokens = prompt_tokens(tok, args.tokens)
    window, memory = raw["fixed_memory_swa_window"], raw["independent_memory"]
    assert tokens.shape[0] > window + memory["max_commit_chunk"], "the prompt must slide the window and commit memory blocks"

    with torch.no_grad():
        taps = forward(cfg, model, memories, tokens, window=window, memory=memory)
    t = tokens.shape[0]
    rows = sorted(set(list(range(0, t, 97)) + [window - 1, window, t - 1]))
    out = {"tokens": tokens.to(torch.int32), "rows": torch.tensor(rows, dtype=torch.int32)}
    for i in memories:
        out[f"layer{i}.mem.refs"] = taps[f"layer{i}.mem.refs"]
        out[f"layer{i}.mem.scores"] = taps[f"layer{i}.mem.scores"]
    for i in range(cfg.num_hidden_layers):
        out[f"layer{i}.out"] = taps[f"layer{i}.out"][rows]
    out["final_hidden"] = taps["final_hidden"][rows]
    out["logits_tail"] = taps["logits"][-8:]
    greedy = taps["logits"][-1].argmax().item()
    print(f"greedy next token after the question: {greedy} {tok.decode([greedy])!r}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "edgerock_real.safetensors")
    save_file({k: v.detach().to(torch.float32).contiguous() for k, v in out.items()}, path)
    manifest = {
        "edgerock_real.safetensors": {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "tensors": {k: list(v.shape) for k, v in out.items()}},
        "_meta": {"T": t, "window": window, "memory": memory, "greedy_next": greedy, "torch_version": torch.__version__, "transformers_version": transformers.__version__},
        "source": source_block(
            checkpoint="rockmatrix-ai/EdgeRock",
            files=files,
            identity={"d_model": cfg.hidden_size, "n_layers": cfg.num_hidden_layers, "n_heads": cfg.num_attention_heads, "n_kv_heads": cfg.num_key_value_heads, "vocab": cfg.vocab_size, "capacity": memory["capacity"], "window": window},
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
