#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reference-parity gates for recurrent-depth
# decoders for its clients. If your team needs expertise in proving a
# prelude/core/coda implementation matches its published reference then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Bake a tiny Huginn reference golden for brain's parity ladder.

Instantiates the released `RavenForCausalLM` at a small shape with a fixed
seed, writes the checkpoint in Hugging Face layout (config + custom modules +
`model.safetensors`), and records what the reference produced for a fixed
latent starting state:

  tokens        [n]        fixed token ids
  input_states  [n, d]     the starting latent state (a fixed tensor, so no RNG)
  logits_k      [n, v]     logits after k recurrent steps, k = 1..T
  latent_k      [n, d]     the latent state after k steps

Driving the reference with an explicit `input_states` is what makes the
comparison exact: its own `like-init` state is a random draw.

usage: huginn_dump_reference.py <huginn_ref_dir> [out_dir]

The released checkpoint is dumped by the same fixed-state procedure:

  huginn_dump_reference.py <ref_dir> <out_dir> --real <hf checkpoint dir> [--steps N]

which runs a fixed eight-token sequence through the real weights and records
the same per-step tensors, for the parity ladder's second rung.
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
    "n_embd": 16,
    "n_heads": 4,
    "n_layers": 4,
    "block_size": 32,
    "vocab_size": 31,
    "padding_multiple": 8,
    "tie_embeddings": True,
    "intermediate_size": 32,
    "bias": False,
    "architecture_class_name": "RecurrentGPT",
    "block_class_name": "SandwichBlock",
    "norm_class_name": "RMSNorm_llama",
    "norm_eps": 1e-6,
    "mlp_class_name": "GatedMLP",
    "nonlin_name": "SiLU",
    "init_strategy": "takase",
    "init_orthogonal": False,
    "state_init": "like-init",
    "injection_type": "linear",
    "n_layers_in_recurrent_block": 2,
    "mean_recurrence": 3,
    "sampling_scheme": "poisson-lognormal-filling",
    "mean_backprop_depth": 1,
    "n_layers_in_prelude": 1,
    "n_layers_in_coda": 1,
    "qk_bias": True,
    "activation_checkpoint_impl": "per-iteration",
    "rope_base": 50000.0,
    "torch_dtype": "float32",
    "transformers_version": "4.44.2",
    "init_values": {"std": 0.02, "embed_scale": 1.0},
}
TOKENS = [3, 5, 7, 11]
STEPS = 3
SEED = 2027


REAL_TOKENS = [3, 5, 7, 11, 13, 17, 19, 23]


def dump_real(ckpt_dir, out, steps):
    """Dump `state_k` / `latent_k` / `logits_k` for the released weights."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
    torch_dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device).eval()
    ids = torch.tensor([REAL_TOKENS], dtype=torch.long, device=device)
    g = torch.Generator().manual_seed(SEED + 1)
    state = torch.randn(1, len(REAL_TOKENS), cfg.n_embd, generator=g).to(device)
    goldens = {
        "tokens": torch.tensor(REAL_TOKENS, dtype=torch.int32),
        "input_states": state.squeeze(0).float().cpu(),
    }
    freqs = model.freqs_cis[:, : ids.shape[1]]
    embeds = model.transformer.wte(ids) * model.emb_scale
    with torch.no_grad():
        for block in model.transformer.prelude:
            embeds = block(embeds, freqs, torch.tensor(0, device=device), None, None)
        for k in range(1, steps + 1):
            x = state.clone()
            for _ in range(k):
                x, _ = model.core_block_forward(x, embeds, freqs, None, None, torch.tensor(-1, device=device), 0)
            out_k = model(
                ids,
                input_states=state,
                num_steps=k,
                use_cache=False,
                output_details={"return_logits": True, "return_latents": True, "return_head": False, "return_stats": False},
            )
            goldens[f"state_{k}"] = x.squeeze(0).float().cpu().contiguous()
            goldens[f"latent_{k}"] = out_k.latent_states.squeeze(0).float().cpu().contiguous()
            goldens[f"logits_{k}"] = out_k.logits.squeeze(0).float().cpu().contiguous()
    save_file(goldens, os.path.join(out, "golden.safetensors"))
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump({
            "model": "huginn",
            "kind": "released",
            "seed": SEED,
            "tokens": REAL_TOKENS,
            "steps": steps,
            "checkpoint": ckpt_dir,
        }, f, indent=2)
    print(f"wrote {out} ({steps} steps, {len(REAL_TOKENS)} tokens)")


def main():
    args = [a for a in sys.argv[1:]]
    if "--real" in args:
        i = args.index("--real")
        ckpt_dir = args[i + 1]
        del args[i : i + 2]
        steps = 2
        if "--steps" in args:
            j = args.index("--steps")
            steps = int(args[j + 1])
            del args[j : j + 2]
        if len(args) != 2:
            sys.exit(__doc__)
        os.makedirs(args[1], exist_ok=True)
        dump_real(ckpt_dir, args[1], steps)
        return
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ref = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "testdata", "golden", "huginn", "tiny")
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    for name in ("raven_config_minimal.py", "raven_modeling_minimal.py"):
        shutil.copy(os.path.join(ref, name), os.path.join(ckpt, name))
    cfg_json = dict(TINY)
    cfg_json.update({
        "architectures": ["RavenForCausalLM"],
        "model_type": "huginn_raven",
        "auto_map": {
            "AutoConfig": "raven_config_minimal.RavenConfig",
            "AutoModelForCausalLM": "raven_modeling_minimal.RavenForCausalLM",
        },
    })
    with open(os.path.join(ckpt, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=2)

    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(SEED)
    cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).float().eval()
    # The head is tied to the embedding table; saving both would write the same
    # rows twice, and brain's importer drops `lm_head.weight` for that reason.
    sd = {k: v.detach().contiguous().float() for k, v in model.state_dict().items() if k not in ("freqs_cis", "lm_head.weight")}
    save_file(sd, os.path.join(ckpt, "model.safetensors"))

    ids = torch.tensor([TOKENS], dtype=torch.long)
    # A fixed starting latent state: no RNG in the comparison.
    g = torch.Generator().manual_seed(SEED + 1)
    state = torch.randn(1, len(TOKENS), cfg.n_embd, generator=g)
    goldens = {
        "tokens": torch.tensor(TOKENS, dtype=torch.int32),
        "input_states": state.squeeze(0).float(),
    }
    with torch.no_grad():
        freqs = model.freqs_cis[:, : ids.shape[1]]
        # The recurrent core re-injects the PRELUDE output, not the embedding.
        embeds = model.transformer.wte(ids) * model.emb_scale
        for block in model.transformer.prelude:
            embeds = block(embeds, freqs, torch.tensor(0), None, None)
        for k in range(1, STEPS + 1):
            # Re-run the reference's own recurrence to capture its raw state.
            x = state.clone()
            for _ in range(k):
                x, _ = model.core_block_forward(x, embeds, freqs, None, None, torch.tensor(-1), 0)
            state_k = x
            out_k = model(
                ids,
                input_states=state,
                num_steps=k,
                use_cache=False,
                output_details={"return_logits": True, "return_latents": True, "return_head": False, "return_stats": False},
            )
            goldens[f"logits_{k}"] = out_k.logits.squeeze(0).float().contiguous()
            goldens[f"latent_{k}"] = out_k.latent_states.squeeze(0).float().contiguous()
            # The raw (pre-`ln_f`) state the next step consumes.
            goldens[f"state_{k}"] = state_k.detach().squeeze(0).float().contiguous()
    save_file(goldens, os.path.join(out, "golden.safetensors"))
    manifest = {
        "model": "huginn",
        "kind": "tiny",
        "seed": SEED,
        "tokens": TOKENS,
        "steps": STEPS,
        "source": source_block(
            checkpoint="tomg-group-umd/huginn-0125 (released raven_modeling_minimal.py, tiny shape)",
            files=[os.path.join(ckpt, "model.safetensors")],
            identity={
                "d_model": cfg.n_embd,
                "n_prelude": cfg.n_layers_in_prelude,
                "n_recurrent": cfg.n_layers_in_recurrent_block,
                "n_coda": cfg.n_layers_in_coda,
                "n_heads": cfg.n_heads,
                "head_dim": cfg.n_embd // cfg.n_heads,
                "d_ff": cfg.intermediate_size,
                "vocab": cfg.vocab_size,
                "steps": STEPS,
                "n_tokens": len(TOKENS),
            },
        ),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out} ({STEPS} steps, {goldens['logits_1'].numel()} logit values per step)")


if __name__ == "__main__":
    main()
