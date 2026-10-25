#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reference-parity gates for looped language
# models for its clients. If your team needs expertise in proving a
# recurrent-depth implementation matches its published reference then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Bake a tiny Ouro reference golden for brain's parity ladder.

Instantiates the released `OuroForCausalLM` at a small shape with a fixed seed,
writes the checkpoint in Hugging Face layout (`config.json` + custom modules +
`model.safetensors`), and records what the reference produced:

  tokens   [n]          fixed token ids
  hidden   [T, n, d]    each pass's post-final-norm hidden state
  gates    [T, n]       each pass's exit-gate logit (pre-sigmoid)
  logits   [T, n, v]    lm_head output for each pass

The tiny shape keeps the golden small enough to regenerate quickly while still
exercising all passes, the gate, the sandwich norms and the RoPE convention.

usage: ouro_dump_reference.py <ouro_reference_dir> [out_dir]

`<ouro_reference_dir>` holds the released `configuration_ouro.py` and
`modeling_ouro.py` (downloaded from the checkpoint repo), which are copied into
the generated checkpoint directory so `trust_remote_code` can load it.
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

TINY = {
    "vocab_size": 29,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 4,
    "max_position_embeddings": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "total_ut_steps": 3,
    "early_exit_threshold": 1.0,
    "use_cache": True,
    "hidden_act": "silu",
    "layer_types": ["full_attention", "full_attention"],
}
TOKENS = [3, 4, 5, 6]
SEED = 4242


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ref = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "golden", "ouro", "tiny")
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    for name in ("configuration_ouro.py", "modeling_ouro.py"):
        shutil.copy(os.path.join(ref, name), os.path.join(ckpt, name))
    cfg_json = dict(TINY)
    cfg_json.update({
        "architectures": ["OuroForCausalLM"],
        "model_type": "ouro",
        "auto_map": {
            "AutoConfig": "configuration_ouro.OuroConfig",
            "AutoModel": "modeling_ouro.OuroModel",
            "AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM",
        },
        "torch_dtype": "float32",
    })
    with open(os.path.join(ckpt, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=2)

    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(SEED)
    cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).float().eval()
    sd = {k: v.detach().contiguous().float() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(ckpt, "model.safetensors"))

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        (_base, hs_list, gate_list) = model.model(input_ids=ids, use_cache=False)
        hidden = torch.stack([h.squeeze(0) for h in hs_list], 0).contiguous()
        gates = torch.stack([g.reshape(-1) for g in gate_list], 0).contiguous()
        logits = torch.stack([model.lm_head(h.squeeze(0)) for h in hs_list], 0).contiguous()

    n_layers = cfg.num_hidden_layers
    assert hidden.shape == (cfg.total_ut_steps, len(TOKENS), cfg.hidden_size), hidden.shape
    assert gates.shape == (cfg.total_ut_steps, len(TOKENS)), gates.shape
    assert logits.shape == (cfg.total_ut_steps, len(TOKENS), cfg.vocab_size), logits.shape

    golden = {
        "tokens": torch.tensor(TOKENS, dtype=torch.int32),
        "hidden": hidden.float(),
        "gates": gates.float(),
        "logits": logits.float(),
    }
    save_file(golden, os.path.join(out, "golden.safetensors"))
    manifest = {
        "model": "ouro",
        "kind": "tiny",
        "seed": SEED,
        "tokens": TOKENS,
        "source": source_block(
            checkpoint="ByteDance/Ouro-2.6B-Thinking (released modeling_ouro.py, tiny shape)",
            files=[os.path.join(ckpt, "model.safetensors")],
            identity={
                "d_model": cfg.hidden_size,
                "n_layers": n_layers,
                "n_heads": cfg.num_attention_heads,
                "n_kv_heads": cfg.num_key_value_heads,
                "head_dim": cfg.head_dim,
                "d_ff": cfg.intermediate_size,
                "vocab": cfg.vocab_size,
                "total_ut_steps": cfg.total_ut_steps,
                "n_tokens": len(TOKENS),
            },
        ),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out} ({hidden.numel()} hidden, {logits.numel()} logit values)")


if __name__ == "__main__":
    main()
