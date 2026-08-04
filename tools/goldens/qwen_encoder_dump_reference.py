#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake a Qwen3-4B penultimate-hidden-state golden for brain's encoder parity.

Z-Image/FLUX.2 feed the DiT `text_encoder(...).hidden_states[-2]` — the output
of the 35th of 36 layers, before the final RMSNorm. This loads the SAME Comfy
single-file weights brain imports, runs a FIXED token sequence through Qwen3
(CPU, fp32), and saves the tokens + `hidden_states[-2]` to a committed fixture.
Fixed token ids (not tokenizer output) isolate forward parity from tokenizer
parity. Dev-time only.
"""
import json, os, sys
from pathlib import Path
import torch
from transformers import Qwen3ForCausalLM, Qwen3Config
from safetensors.torch import load_file, save_file

if len(sys.argv) < 3:
    sys.exit("usage: qwen_encoder_dump_reference.py <enc_config_dir> <weights.safetensors> [out.safetensors]\n"
             "  <enc_config_dir>: dir with the Qwen3-4B config.json; <weights>: the encoder safetensors")
ENC_DIR = sys.argv[1]
WEIGHTS = sys.argv[2]
# testdata/golden/qwen/... by default -- where
# crates/qwen/tests/encoder_parity.rs's testdata("golden/qwen/qwen3_4b_encoder.safetensors")
# actually looks.
_TESTDATA = os.environ.get("BRAIN_TESTDATA") or str(Path(__file__).resolve().parents[2] / "testdata")
OUT = sys.argv[3] if len(sys.argv) > 3 else str(Path(_TESTDATA) / "golden" / "qwen" / "qwen3_4b_encoder.safetensors")

# A fixed, arbitrary token sequence (valid ids < vocab 151936).
TOKENS = [9707, 11, 419, 374, 264, 2613, 1273, 13]

def main():
    cfg = Qwen3Config.from_pretrained(ENC_DIR)
    model = Qwen3ForCausalLM(cfg)
    sd = load_file(WEIGHTS)  # HF keys ("model.*"), tied lm_head absent
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Only the tied lm_head may be "missing"; nothing should be unexpected.
    assert all("lm_head" in m for m in missing), f"unexpected missing: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    model = model.float().eval()

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states[-2].squeeze(0).contiguous()  # [T, 2560]
    n_hs = len(out.hidden_states)
    print(f"hidden_states tuple len={n_hs} (n_layers+1); using index -2 = layer {n_hs-2-1} output")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        {"tokens": torch.tensor(TOKENS, dtype=torch.int32),
         "hidden": hs.to(torch.float32)},
        OUT,
        metadata={"src": "Qwen3-4B hidden_states[-2] (penultimate)", "hidden_size": str(hs.shape[-1])},
    )
    print(f"wrote {OUT}  hidden {tuple(hs.shape)}  "
          f"[min,max,mean]=[{hs.min():.4f},{hs.max():.4f},{hs.mean():.4f}]")

if __name__ == "__main__":
    sys.exit(main())
