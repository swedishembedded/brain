#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Real-weight greedy-generation golden for Qwen3-Omni-30B-A3B-Instruct's
Thinker text decoder - deliberately SEPARATE from `qwen3omnimoe_dump_reference.py`,
whose own header commits to "component-scoped only, the full checkpoint is
too large to load wholesale even once." This script does exactly that
(loads the whole Thinker, all 48 layers / 128 experts, real weights) and is
opt-in for that reason: only run it when you actually want the M9
generation-loop validation, not as part of the regular component-parity
sweep.

Uses `Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(...,
torch_dtype=bfloat16, low_cpu_mem_usage=True)` rather than the manual
state_dict/`.float()` loading `qwen3omnimoe_dump_reference.py`'s component dumps use:
`from_pretrained` streams each shard's tensors directly into a meta-device-
initialized model (no full-fp32-random-init pass, no double allocation), and
`base_model_prefix = "thinker"` on this class makes it match the checkpoint's
`thinker.*`-prefixed keys against its own unprefixed state dict automatically
-- `talker.*`/`code2wav.*`/`talker.code_predictor.*` keys are reported as
"unexpected" (harmless) and never loaded. bf16 keeps Thinker's ~30B params at
~60GB resident instead of ~120GB in fp32.

No sampling: greedy (argmax) generation only, matching brain's own
`omni::generate::generate_greedy`, so a byte-exact token-id match is the
correctness bar (float rounding at the very edge of a near-tied logit pair
could theoretically flip the argmax, but everything downstream has already
been validated to cosine 1.000000, so this is a loop-control check, not a
new numerics check).

usage: qwen3omnimoe_dump_generate.py <hf_checkpoint_dir> [out_dir] [--max-new N]
"""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from safetensors.torch import save_file

if len(sys.argv) < 2:
    sys.exit(__doc__)
HF_DIR = sys.argv[1]
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else os.path.join(
    os.path.dirname(__file__), "..", "..", "testdata", "golden", "omni")
os.makedirs(OUT_DIR, exist_ok=True)
MAX_NEW = 5
for a in sys.argv[2:]:
    if a.startswith("--max-new="):
        MAX_NEW = int(a.split("=", 1)[1])

torch.manual_seed(0)


def main():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

    import json
    with open(os.path.join(HF_DIR, "config.json")) as f:
        raw = json.load(f)
    full_cfg = Qwen3OmniMoeConfig.from_dict(raw)

    print("loading Thinker (bf16, low_cpu_mem_usage) -- this streams ~60GB, expect several minutes...", flush=True)
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        HF_DIR, config=full_cfg.thinker_config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    print("loaded.", flush=True)

    # Fixed, short, plain-text prompt -- no chat template, no image/audio, so
    # this is a pure test of layer-chaining + sampling + tokenizer round-trip
    # against the SAME diagonal-M-RoPE-collapse case M6/M7's layer0 tests use.
    prompt_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13]  # same fixed ids as the layer0 golden
    ids = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    generated = out[0].tolist()
    print("generated ids:", generated, flush=True)

    tensors = {
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.int32),
        "generated_ids": torch.tensor(generated, dtype=torch.int32),
    }
    out_path = os.path.join(OUT_DIR, "omni_generate.safetensors")
    save_file(tensors, out_path, metadata={"max_new_tokens": str(MAX_NEW), "src": "thinker (full, real weights)"})
    print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
