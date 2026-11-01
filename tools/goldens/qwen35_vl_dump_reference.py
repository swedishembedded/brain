#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements validation of vision-language checkpoints against their reference implementation for
# its clients. If your team needs expertise in proving that a served multimodal model reads a picture as its reference
# does then you can procure our services by sending an email to info@swedishembedded.com.
"""Dump a REAL-WEIGHT reference golden for Qwen3.8-27B reading one small image: the vision tower's output, the
three-axis rotary positions of the prompt, and the decoder's logits for the first generated tokens.

The reference is `transformers.models.qwen3_5.Qwen3_5ForConditionalGeneration` built from the released FP8 checkpoint
with every FP8 linear decoded to bf16 (`value * block scale`, the arithmetic a bf16 server of the release runs), the
vision tower held in float32 so that the tower's own numbers are the tight half of the comparison. Greedy generation
supplies the continuation (HF's own `rope_deltas` handling for the tokens after the image).

The image must already be a size the processor would resize to (both sides multiples of 32, inside the checkpoint's
pixel bounds): the golden then pins the tower, the patch packing, the template and the positions without depending on
a resampler. The packing replicates `Qwen2VLImageProcessorFast.patchify` of the installed transformers (read, not
imported: importing it needs torchvision).

Needs `PIL` besides the torch environment (`PYTHONPATH` may point at a site-packages that has it).

Usage:
  python tools/goldens/qwen35_vl_dump_reference.py --checkpoint DIR --image PNG [--out DIR] [--steps 3]
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
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

QUESTION = "Describe the part and list every dimension written on it."


def load_checkpoint(model, ckpt_dir):
    """Fill `model` from the FP8 release: every `weight` with a `weight_scale_inv` is E4M3 bytes times its 128 x 128
    block scale, rounded to bf16; everything else is stored bf16. `mtp.*` (the draft head) is not part of HF's model."""
    index = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))["weight_map"]
    state = model.state_dict()
    wanted = set(state.keys())
    seen = set()
    files = sorted(set(index.values()))
    handles = {f: safe_open(os.path.join(ckpt_dir, f), "pt") for f in files}
    for name, f in index.items():
        if name.startswith("mtp.") or name.endswith("weight_scale_inv"):
            continue
        t = handles[f].get_tensor(name)
        scale_name = name + "_scale_inv"
        if scale_name in index:
            s = handles[index[scale_name]].get_tensor(scale_name).float()
            rows, cols = t.shape
            s = s.repeat_interleave(128, dim=0)[:rows].repeat_interleave(128, dim=1)[:, :cols]
            t = (t.float() * s).to(torch.bfloat16)
        if name not in wanted:
            raise SystemExit(f"checkpoint tensor {name} has no counterpart in the HF model")
        with torch.no_grad():
            state[name].copy_(t.to(state[name].dtype))
        seen.add(name)
    missing = wanted - seen
    if missing:
        raise SystemExit(f"{len(missing)} HF tensors were not in the checkpoint, e.g. {sorted(missing)[:5]}")


def patchify(img, cfg):
    """`Qwen2VLImageProcessorFast.patchify` for one normalized CHW image: [patches, C * temporal * patch * patch]."""
    vc = cfg.vision_config
    p, m, tps = vc.patch_size, vc.spatial_merge_size, vc.temporal_patch_size
    c, h, w = img.shape
    gh, gw = h // p, w // p
    x = img.reshape(1, c, gh // m, m, p, gw // m, m, p).permute(0, 2, 5, 3, 6, 1, 4, 7)
    x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, tps, -1, -1)
    return x.reshape(gh * gw, c * tps * p * p).contiguous(), gh, gw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the Qwen3.8-27B-FP8 directory")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=os.path.join("testdata", "golden", "qwen35", "vl"))
    ap.add_argument("--steps", type=int, default=3, help="tokens generated after the prompt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    cfg = AutoConfig.from_pretrained(args.checkpoint)
    cfg.quantization_config = None
    pre = json.load(open(os.path.join(args.checkpoint, "preprocessor_config.json")))

    # The processor's normalization on a picture that needs no resize.
    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    factor = pre["patch_size"] * pre["merge_size"]
    assert h % factor == 0 and w % factor == 0, f"{w}x{h} is not a multiple of {factor}: resize the image first"
    assert pre["size"]["shortest_edge"] <= h * w <= pre["size"]["longest_edge"]
    arr = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(pre["image_mean"]).view(3, 1, 1)
    std = torch.tensor(pre["image_std"]).view(3, 1, 1)
    patches, gh, gw = patchify((arr - mean) / std, cfg)
    grid_thw = torch.tensor([[1, gh, gw]], dtype=torch.long)
    n_tokens = gh * gw // (cfg.vision_config.spatial_merge_size**2)

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    assert text.count(placeholder) == 1, text
    text = text.replace(placeholder, "<|vision_start|>" + "<|image_pad|>" * n_tokens + "<|vision_end|>")
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"image {w}x{h}: grid {gh}x{gw} patches, {n_tokens} image tokens, prompt {ids.shape[1]} tokens")

    print("building the HF model (bf16) from the FP8 release ...")
    torch.set_default_dtype(torch.bfloat16)
    with no_init_weights():
        model = Qwen3_5ForConditionalGeneration(cfg)
    torch.set_default_dtype(torch.float32)
    model.eval()
    load_checkpoint(model, args.checkpoint)
    # The tower in float32: its numbers are the tight half of the comparison, and the decoder sees its rows rounded to
    # bf16 as the HF model rounds them.
    visual = model.model.visual.float()

    pixel_values = patches.float()
    out = visual(pixel_values, grid_thw=grid_thw)
    hidden, merged = out.last_hidden_state, out.pooler_output
    assert merged.shape == (n_tokens, cfg.text_config.hidden_size), merged.shape

    mm_token_type_ids = (ids == cfg.image_token_id).to(torch.int32)
    position_ids, rope_deltas = model.model.get_rope_index(ids, mm_token_type_ids, image_grid_thw=grid_thw)

    gen = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        max_new_tokens=args.steps,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    new_tokens = gen.sequences[0, ids.shape[1]:]
    print("generated:", new_tokens.tolist(), repr(tok.decode(new_tokens)))

    tensors = {
        "input_ids": ids[0].to(torch.int32),
        "patches": patches,
        "grid_thw": grid_thw.to(torch.int32),
        "hidden": hidden.float().contiguous(),
        "merged": merged.float().contiguous(),
        "position_ids": position_ids[:, 0, :].to(torch.int32).contiguous(),
        "rope_delta": rope_deltas.reshape(-1).to(torch.int32),
        "new_tokens": new_tokens.to(torch.int32),
    }
    for i, s in enumerate(gen.scores):
        tensors[f"logits{i}"] = s[0].float().contiguous()
    path = os.path.join(args.out, "qwen35_vl.safetensors")
    save_file({k: v.clone() for k, v in tensors.items()}, path)
    manifest = {
        "qwen35_vl.safetensors": {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "tensors": {k: list(v.shape) for k, v in tensors.items()}},
        "_meta": {"question": QUESTION, "image": os.path.basename(args.image), "size": [w, h], "steps": args.steps, "torch_version": torch.__version__, "transformers_version": transformers.__version__},
        "source": source_block(
            checkpoint="Qwen/Qwen3.8-27B-FP8",
            files=[os.path.join(args.checkpoint, "model.safetensors.index.json")],
            identity={"hidden_size": cfg.text_config.hidden_size, "num_hidden_layers": cfg.text_config.num_hidden_layers, "vocab_size": cfg.text_config.vocab_size, "vision_depth": cfg.vision_config.depth},
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
