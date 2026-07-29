#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Bake staged LFM2.5-Encoder goldens for brain's parity ladder.

Loads the released checkpoint through the repo's OWN
`modeling_lfm2_bidirectional.py` (the encoder patches: bidirectional mask +
non-causal short-conv), runs a FIXED token sequence (CPU, fp32), and saves per
stage: the post-embedding residual, every layer output, the final hidden state
(post embedding_norm), MLM logits at three probe rows, and a fill-mask top-5.
Fixed token ids isolate forward parity from tokenizer parity. Dev-time only.

usage: lfm_dump_reference.py <hf_checkpoint_dir> [out.safetensors]
"""
import importlib.util
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

if len(sys.argv) < 2:
    sys.exit(__doc__)
HF_DIR = sys.argv[1]
NAME = os.path.basename(HF_DIR.rstrip("/")).lower().replace(".", "").replace("-", "_")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(__file__), "..", "crates", "lfm", "tests", "golden", f"{NAME}.safetensors")

# A fixed, arbitrary token sequence (valid ids < vocab 65536); id 16 = <|mask|>.
# "<|startoftext|>The capital of France is<|mask|>." tokenized by HF tokenizers.
TOKENS = [1, 1098, 5706, 803, 4481, 856, 16, 523]
LOGIT_ROWS = [0, 6, 7]  # BOS row, the <|mask|> row, the last row


def load_model(hf_dir):
    spec = importlib.util.spec_from_file_location(
        "modeling_lfm2_bidirectional", os.path.join(hf_dir, "modeling_lfm2_bidirectional.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

    cfg = Lfm2Config.from_pretrained(hf_dir)
    model = mod.Lfm2BidirectionalForMaskedLM(cfg)
    sd = load_file(os.path.join(hf_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert all("lm_head" in m for m in missing), f"unexpected missing: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    return model.float().eval()


def main():
    model = load_model(HF_DIR)
    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states  # embeddings + each layer output (pre final norm)
    n_layers = model.config.num_hidden_layers
    assert len(hs) == n_layers + 1, f"hidden_states len {len(hs)} != {n_layers + 1}"

    with torch.no_grad():
        hidden = model.lfm2(ids).last_hidden_state  # post embedding_norm
    logits = out.logits.squeeze(0)

    tensors = {
        "tokens": torch.tensor(TOKENS, dtype=torch.int32),
        "logit_rows": torch.tensor(LOGIT_ROWS, dtype=torch.int32),
        "hidden": hidden.squeeze(0).contiguous().to(torch.float32),
        "logits_probe": logits[LOGIT_ROWS].contiguous().to(torch.float32),
    }
    for l, h in enumerate(hs):
        tensors[f"res{l}"] = h.squeeze(0).contiguous().to(torch.float32)

    mask_row = TOKENS.index(16)
    top5 = torch.topk(logits[mask_row], 5)
    tensors["mask_top5_ids"] = top5.indices.to(torch.int32)
    tensors["mask_top5_logits"] = top5.values.to(torch.float32)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata={
        "src": f"{os.path.basename(HF_DIR)} modeling_lfm2_bidirectional fp32 CPU",
        "n_layers": str(n_layers),
    })
    print(f"wrote {OUT}: {len(tensors)} tensors, hidden {tuple(hidden.shape)}, "
          f"mask top5 ids {top5.indices.tolist()}")


if __name__ == "__main__":
    sys.exit(main())
