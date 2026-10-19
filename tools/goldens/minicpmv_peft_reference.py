#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for carrying fine-tuned multimodal
# models from one training stack to another for its clients. If your team
# needs expertise in vision-language model fine-tuning then you can procure
# our services by sending an email to info@swedishembedded.com.
"""Golden for `crates/minicpmv`'s PEFT export: what `transformers` computes
with brain's exported adapter applied to the real `openbmb/MiniCPM-V-4.6`.

`crates/minicpmv/tests/peft_export.rs` writes a deterministic tune's PEFT
export to `<out>/adapter/` (decoder LoRA, tower LoRA at a different rank and
scale, the connector trained in full). This script applies it to the HF model
in fp32 on CPU and records, for one synthetic image and question:

  peft/image.png              the image, so the Rust side decodes the same bytes
  peft/logits.safetensors     `logits_last` (the next-token logits after the
                              prompt), `input_ids` and the `greedy`
                              continuation
  peft/manifest.json          the adapter file's sha256 (the test refuses a
                              golden of another adapter), how the adapter was
                              applied, versions and the `source` block

The adapter goes through `peft.PeftModel` when `peft` is installed. Without it
the script applies the adapter exactly as PEFT's LoRA layer computes it -
`W += (lora_alpha / r) * B @ A` on every `target_modules` entry and each
`modules_to_save` tensor loaded in place - and records `"applied": "manual"`.

Usage:
  python tools/goldens/minicpmv_peft_reference.py --model <dir> [--out DIR]
"""

import argparse
import hashlib
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import transformers
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
QUESTION = "Describe this image in one sentence."
GREEDY_TOKENS = 16
PREFIX = "base_model.model."


def picture():
    """A 336x252 card: gradients, a fine interference texture and a disc (no
    RNG; the saved PNG is what the Rust side reads either way)."""
    h, w = 252, 336
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    texture = (np.sin(xx * 0.71) * np.cos(yy * 0.53) + 1.0)[..., None] * 30.0
    img = np.stack([xx / (w - 1), yy / (h - 1), 1.0 - xx / (w - 1)], -1) * 180.0 + texture
    img[(xx - 200) ** 2 + (yy - 110) ** 2 < 60 ** 2] = [230, 40, 40]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def apply_manually(model, adapter_dir):
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    modules = dict(model.named_modules())
    scale = cfg["lora_alpha"] / cfg["r"]
    with torch.no_grad():
        for name in cfg["target_modules"]:
            a = tensors.pop(f"{PREFIX}{name}.lora_A.weight")
            b = tensors.pop(f"{PREFIX}{name}.lora_B.weight")
            modules[name].weight += scale * (b @ a)
        params = dict(model.named_parameters())
        for key in list(tensors):
            name = key[len(PREFIX):]
            if not any(name.startswith(m + ".") for m in cfg.get("modules_to_save", [])):
                raise SystemExit(f"{key} is neither a LoRA factor of a target module nor a module to save")
            p = params[name]
            p.copy_(tensors.pop(key).reshape(p.shape))
    if tensors:
        raise SystemExit(f"unused adapter tensors: {sorted(tensors)[:5]}")
    return model


def apply_adapter(model, adapter_dir):
    try:
        import peft
    except ImportError:
        return apply_manually(model, adapter_dir), "manual"
    return peft.PeftModel.from_pretrained(model, adapter_dir).merge_and_unload(), f"peft {peft.__version__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local openbmb/MiniCPM-V-4.6 snapshot directory")
    ap.add_argument("--out", default=os.path.join(REPO, "testdata", "minicpmv", "peft"))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    adapter_dir = os.path.join(args.out, "adapter")
    adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(adapter_file):
        raise SystemExit(f"{adapter_file} is missing - run `cargo test -p brain-minicpmv --test peft_export` first, it writes the adapter")

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float32).eval()
    model, applied = apply_adapter(model, adapter_dir)

    image = picture()
    image.save(os.path.join(args.out, "image.png"))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    inp = proc(text=[text], images=[image], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits[0, -1].float()
        gen = model.generate(**inp, max_new_tokens=GREEDY_TOKENS, do_sample=False)
    ids = inp["input_ids"][0]
    cont = gen[0, ids.shape[0]:]
    save_file(
        {"logits_last": logits.contiguous(), "input_ids": ids.float(), "greedy": cont.float()},
        os.path.join(args.out, "logits.safetensors"),
    )
    cfg = model.config
    tc, vc = cfg.text_config, cfg.vision_config
    weights = sorted(os.path.join(args.model, f) for f in os.listdir(args.model) if f.endswith(".safetensors"))
    manifest = {
        "adapter_sha256": hashlib.sha256(open(adapter_file, "rb").read()).hexdigest(),
        "applied": applied,
        "question": QUESTION,
        "greedy_text": proc.tokenizer.decode(cont, skip_special_tokens=False),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "source": source_block(
            checkpoint="openbmb/MiniCPM-V-4.6",
            files=weights,
            identity={
                "text_n_layers": tc.num_hidden_layers, "text_d_model": tc.hidden_size, "vocab_size": tc.vocab_size,
                "vision_n_layers": vc.num_hidden_layers, "vision_d_model": vc.hidden_size, "insert_layer_id": cfg.insert_layer_id,
            },
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"applied: {applied}; greedy: {manifest['greedy_text']!r}")


if __name__ == "__main__":
    main()
