#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the DeepSeek-VL-7B-chat end-to-end reference for `crates/deepseekvl`.

The towers have their own goldens (`deepseek_vl_sam_dump_reference.py`,
`siglip_dump_reference.py`); this one covers what joins them: preprocessing,
the hybrid tower's resize, the split aligner, the prompt template and image
splice, the decoder's logits and a greedy continuation. It runs the pinned
reference (`deepseek_vl`, imported from a DeepSeek-VL checkout) in fp32 on the
CPU over a fixed synthetic non-square image, and writes:

  golden.safetensors   every tensor as f32:
                         image          [H, W, 3]      the RGB8 input, as values
                         pixel_values   [3, 1024, 1024] processor output, [0, 1]
                         low_images     [3, 384, 384]  the low branch's resize
                         high_features  [576, 1024]    SAM branch (rows = cells)
                         low_features   [576, 1024]    SigLIP branch
                         aligner_out    [576, 4096]    the split MLP aligner
                         inputs_embeds  [T, 4096]      text embeddings, image spliced
                         logits_last    [vocab]        the prompt's next-token logits
  manifest.json        the rendered prompt, its token ids, the greedy ids, the
                       image-token id, shapes, sha256s and the source block.

The DeepSeek-VL venv (torch 2.0.1) cannot hand tensors to numpy, so tensors
leave through `safetensors.torch` only.

Usage:
  python tools/goldens/deepseekvl_dump_reference.py \\
      --src <DeepSeek-VL checkout> \\
      --ckpt <deepseek-ai/deepseek-vl-7b-chat snapshot dir> \\
      --out "$BRAIN_TESTDATA/deepseek-vl/composite"
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

CHECKPOINT = "deepseek-ai/deepseek-vl-7b-chat"
IMAGE_W, IMAGE_H = 320, 200
QUESTION = "<image_placeholder>Describe this image."
GREEDY = 16


def synthetic_image():
    """A deterministic RGB8 image with structure a resampler cannot fake:
    smooth gradients, a hard-edged disc and a checker band."""
    y = torch.arange(IMAGE_H, dtype=torch.float32).view(-1, 1).expand(IMAGE_H, IMAGE_W)
    x = torch.arange(IMAGE_W, dtype=torch.float32).view(1, -1).expand(IMAGE_H, IMAGE_W)
    r = 255.0 * x / (IMAGE_W - 1)
    g = 255.0 * y / (IMAGE_H - 1)
    b = 128.0 + 100.0 * torch.sin(x / 17.0) * torch.cos(y / 11.0)
    disc = ((x - 210.0) ** 2 + (y - 90.0) ** 2) < 55.0**2
    r = torch.where(disc, torch.full_like(r, 230.0), r)
    g = torch.where(disc, torch.full_like(g, 40.0), g)
    b = torch.where(disc, torch.full_like(b, 30.0), b)
    checker = (y > 160) & (((x // 10) + (y // 10)) % 2 == 0)
    rgb = torch.stack([r, g, b], dim=-1)
    rgb = torch.where(checker.unsqueeze(-1), torch.zeros_like(rgb), rgb)
    return rgb.round().clamp(0, 255).to(torch.uint8)


def source_commit(src):
    try:
        return subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.environ.get("DEEPSEEK_VL_SRC"), help="DeepSeek-VL checkout (the pinned reference source)")
    ap.add_argument("--ckpt", required=True, help="deepseek-vl-7b-chat snapshot directory")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    if not args.src:
        sys.exit("--src (or DEEPSEEK_VL_SRC) must name a DeepSeek-VL checkout")
    sys.path.insert(0, args.src)
    from PIL import Image  # noqa: E402

    from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor  # noqa: E402

    torch.set_grad_enabled(False)
    processor = VLChatProcessor.from_pretrained(args.ckpt)
    model = MultiModalityCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float32).float().eval()

    rgb = synthetic_image()
    image = Image.frombytes("RGB", (IMAGE_W, IMAGE_H), bytes(rgb.flatten().tolist()))
    conversation = [{"role": "User", "content": QUESTION, "images": [image]}, {"role": "Assistant", "content": ""}]
    inputs = processor(conversations=conversation, images=[image], force_batchify=True)

    pixel_values = inputs.pixel_values.float()
    tower = model.vision_model
    low_images = tower.resize(pixel_values[:, 0])
    high, low = tower(pixel_values[:, 0])
    aligner_out = model.aligner((high, low))
    embeds = model.prepare_inputs_embeds(**inputs)
    logits = model.language_model(inputs_embeds=embeds, attention_mask=inputs.attention_mask).logits
    generated = model.language_model.generate(
        inputs_embeds=embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=GREEDY,
        min_new_tokens=GREEDY,
        do_sample=False,
        use_cache=True,
    )
    greedy = generated[0].tolist()[-GREEDY:]
    print("prompt:", repr(inputs.sft_format[0]))
    print("greedy:", greedy, repr(processor.tokenizer.decode(greedy)))

    tensors = {
        "image": rgb.float(),
        "pixel_values": pixel_values[0, 0],
        "low_images": low_images[0],
        "high_features": high[0],
        "low_features": low[0],
        "aligner_out": aligner_out[0],
        "inputs_embeds": embeds[0],
        "logits_last": logits[0, -1],
    }
    tensors = {k: v.detach().float().contiguous() for k, v in tensors.items()}
    os.makedirs(args.out, exist_ok=True)
    golden = os.path.join(args.out, "golden.safetensors")
    save_file(tensors, golden)

    lang = model.language_model.config
    manifest = {
        "dumper": "tools/goldens/deepseekvl_dump_reference.py",
        "reference": {"module": "deepseek_vl.models.MultiModalityCausalLM", "commit": source_commit(args.src)},
        "dtype": "float32",
        "device": "cpu",
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "prompt": inputs.sft_format[0],
        "input_ids": inputs.input_ids[0].tolist(),
        "image_token_id": processor.image_id,
        "greedy_ids": greedy,
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
        "files": {"golden.safetensors": "sha256:" + sha256_file(golden)},
        "source": source_block(
            checkpoint=CHECKPOINT,
            files=[os.path.join(args.ckpt, "config.json")],
            hash_files=True,
            identity={"n_layers": lang.num_hidden_layers, "d_model": lang.hidden_size, "vocab": lang.vocab_size, "image_tokens": processor.num_image_tokens},
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {golden}")


if __name__ == "__main__":
    main()
