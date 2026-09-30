#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the Janus-Pro-7B reference for `crates/januspro`.

Runs the pinned reference (`janus`, imported from a Janus checkout) in fp32
on the CPU and writes two goldens. The VQ-16 image tokenizer has its own
(`janus_vq16_dump_reference.py`).

understanding/  the chat path over the same fixed synthetic image
                `deepseekvl_dump_reference.py` uses:
  golden.safetensors   image [H, W, 3], pixel_values [3, 384, 384],
                       features [576, 1024] (SigLIP), aligner_out [576, 4096],
                       inputs_embeds [T, 4096], logits_last [vocab]
  manifest.json        prompt, input_ids, greedy_ids, image/boi/eoi ids

generation/     classifier-free-guided text-to-image, teacher-forced: the
                reference samples STEPS image tokens for one image (a
                conditional and an unconditional row), and records at every
                step what a port must reproduce given the same tokens.
  golden.safetensors   logits_cond [STEPS, 16384], logits_uncond [STEPS, 16384]
                       (gen_head outputs), logits_cfg [STEPS, 16384] (the
                       guided blend), token_embeds [STEPS, 4096]
                       (gen_aligner(gen_embed(token)) fed back)
  manifest.json        prompt, cond_ids, uncond_ids, sampled ids, cfg_weight,
                       seed

The multimodal venv (torch 2.0.1) cannot hand tensors to numpy, so tensors
leave through `safetensors.torch` only.

Usage:
  python tools/goldens/janus_dump_reference.py \\
      --src <Janus checkout> --ckpt <deepseek-ai/Janus-Pro-7B snapshot dir> \\
      --out "$BRAIN_TESTDATA/janus"
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
from deepseekvl_dump_reference import synthetic_image, IMAGE_W, IMAGE_H  # noqa: E402

CHECKPOINT = "deepseek-ai/Janus-Pro-7B"
QUESTION = "<image_placeholder>\nDescribe this image."
GREEDY = 16
GEN_PROMPT = "A red apple on a wooden table."
CFG_WEIGHT = 5.0
STEPS = 8
SEED = 20260930


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


def write(out, tensors, manifest, args, model):
    os.makedirs(out, exist_ok=True)
    tensors = {k: v.detach().float().contiguous() for k, v in tensors.items()}
    golden = os.path.join(out, "golden.safetensors")
    save_file(tensors, golden)
    lang = model.language_model.config
    manifest.update(
        {
            "dumper": "tools/goldens/janus_dump_reference.py",
            "reference": {"module": "janus.models.MultiModalityCausalLM", "commit": source_commit(args.src)},
            "dtype": "float32",
            "device": "cpu",
            "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
            "tensors": {k: list(v.shape) for k, v in tensors.items()},
            "files": {"golden.safetensors": "sha256:" + sha256_file(golden)},
            "source": source_block(
                checkpoint=CHECKPOINT,
                files=[os.path.join(args.ckpt, "config.json")],
                hash_files=True,
                identity={"n_layers": lang.num_hidden_layers, "d_model": lang.hidden_size, "vocab": lang.vocab_size, "image_vocab": model.gen_head.vision_head.out_features},
            ),
        }
    )
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {golden}")


def understanding(processor, model, args):
    from PIL import Image  # noqa: E402

    rgb = synthetic_image()
    image = Image.frombytes("RGB", (IMAGE_W, IMAGE_H), bytes(rgb.flatten().tolist()))
    conversation = [{"role": "<|User|>", "content": QUESTION, "images": [image]}, {"role": "<|Assistant|>", "content": ""}]
    inputs = processor(conversations=conversation, images=[image], force_batchify=True)
    pixel_values = inputs.pixel_values.float()
    features = model.vision_model(pixel_values[:, 0])
    aligner_out = model.aligner(features)
    embeds = model.prepare_inputs_embeds(**inputs)
    logits = model.language_model(inputs_embeds=embeds, attention_mask=inputs.attention_mask).logits
    tok = processor.tokenizer
    generated = model.language_model.generate(
        inputs_embeds=embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        max_new_tokens=GREEDY,
        min_new_tokens=GREEDY,
        do_sample=False,
        use_cache=True,
    )
    greedy = generated[0].tolist()[-GREEDY:]
    print("understanding prompt:", repr(inputs.sft_format[0]))
    print("greedy:", greedy, repr(tok.decode(greedy)))
    tensors = {
        "image": rgb.float(),
        "pixel_values": pixel_values[0, 0],
        "features": features[0],
        "aligner_out": aligner_out[0],
        "inputs_embeds": embeds[0],
        "logits_last": logits[0, -1],
    }
    manifest = {
        "prompt": inputs.sft_format[0],
        "input_ids": inputs.input_ids[0].tolist(),
        "image_token_id": processor.image_id,
        "image_start_id": processor.image_start_id,
        "image_end_id": processor.image_end_id,
        "greedy_ids": greedy,
    }
    write(os.path.join(args.out, "understanding"), tensors, manifest, args, model)


def generation(processor, model, args):
    conversation = [{"role": "<|User|>", "content": GEN_PROMPT}, {"role": "<|Assistant|>", "content": ""}]
    sft = processor.apply_sft_template_for_multi_turn_prompts(conversations=conversation, sft_format=processor.sft_format, system_prompt="")
    prompt = sft + processor.image_start_tag
    cond = torch.LongTensor(processor.tokenizer.encode(prompt))
    uncond = cond.clone()
    uncond[1:-1] = processor.pad_id
    tokens = torch.stack([cond, uncond])
    embeds = model.language_model.get_input_embeddings()(tokens)
    gen = torch.Generator().manual_seed(SEED)
    rows = {"logits_cond": [], "logits_uncond": [], "logits_cfg": [], "token_embeds": []}
    sampled = []
    past = None
    for _ in range(STEPS):
        out = model.language_model.model(inputs_embeds=embeds, use_cache=True, past_key_values=past)
        past = out.past_key_values
        logits = model.gen_head(out.last_hidden_state[:, -1, :])
        blended = logits[1] + CFG_WEIGHT * (logits[0] - logits[1])
        probs = torch.softmax(blended, dim=-1)
        token = torch.multinomial(probs, num_samples=1, generator=gen)
        fed = model.prepare_gen_img_embeds(token.repeat(2))
        rows["logits_cond"].append(logits[0])
        rows["logits_uncond"].append(logits[1])
        rows["logits_cfg"].append(blended)
        rows["token_embeds"].append(fed[0])
        sampled.append(int(token))
        embeds = fed.unsqueeze(1)
    print("generation prompt:", repr(prompt), "sampled:", sampled)
    tensors = {k: torch.stack(v) for k, v in rows.items()}
    manifest = {
        "prompt": prompt,
        "cond_ids": cond.tolist(),
        "uncond_ids": uncond.tolist(),
        "pad_id": processor.pad_id,
        "sampled": sampled,
        "cfg_weight": CFG_WEIGHT,
        "seed": SEED,
    }
    write(os.path.join(args.out, "generation"), tensors, manifest, args, model)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.environ.get("JANUS_SRC"), help="Janus checkout (the pinned reference source)")
    ap.add_argument("--ckpt", required=True, help="Janus-Pro-7B snapshot directory")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    if not args.src:
        sys.exit("--src (or JANUS_SRC) must name a Janus checkout")
    sys.path.insert(0, args.src)
    from janus.models import MultiModalityCausalLM, VLChatProcessor  # noqa: E402

    torch.set_grad_enabled(False)
    processor = VLChatProcessor.from_pretrained(args.ckpt)
    model = MultiModalityCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.float32).float().eval()
    understanding(processor, model, args)
    generation(processor, model, args)


if __name__ == "__main__":
    main()
