#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a tiny-config golden for MiniMax-H3's ref2va PRESENTATION encoder.

A ref2va request reaches the Qwen3-VL conditioner as one presentation: a
label per reference (`"<Picture i>: "` plus an image vision block, `"<Audio
j>: "` alone, `"<Video k>: "` plus one timestamped vision block per merged
frame pair), then the prompt. This dumper runs the installed diffusers'
`MiniMaxH3Ref2VATextEncoderStep` helpers - frame sampling, the per-modality
processor calls, the presentation tokenizer - and `get_qwen3vl_prompt_embeds`
over a TINY random Qwen3-VL (the real one is 33B-class) built with the real
checkpoint's architecture switches (interleaved M-RoPE, DeepStack, tanh-GELU
ViT, untied head) and the REAL tokenizer and processor configs, so the token
stream, the vision patches, the rotary layout and the DeepStack adds are all
the real ones at a small width.

Writes into --out:
  text_encoder/config.json, text_encoder/model.safetensors   the tiny model
  minimaxh3_presentation.safetensors
    {image,video}_visual, {image,video}_deepstack_{0,1,2}
                          [rows, hidden] the vision tower's own outputs
    image                 [H, W, 3]     the image reference (uint8 as f32)
    video                 [T, H, W, 3]  the video reference at 24 fps
    token_ids, token_tags [n]
    hidden                [n, hidden]   hidden_states[LAYER]
  manifest.json

Usage:
  python3 tools/goldens/minimaxh3_presentation_dump_reference.py \\
      --h3-dir /path/to/MiniMaxAI/MiniMax-H3 \\
      --out testdata/golden/minimaxh3/presentation --seed 7
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

from diffusers.modular_pipelines.minimax_h3.encoders import (  # noqa: E402
    MiniMaxH3Ref2VATextEncoderStep,
    get_qwen3vl_prompt_embeds,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (  # noqa: E402
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)
from diffusers.modular_pipelines.minimax_h3.references import (  # noqa: E402
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)

LAYER = 3
FPS = 24.0
PROMPT = "A red kite rises over the dunes while a drum keeps time."


def tiny_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=LAYER + 1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-6,
            vocab_size=151936,
            rope_theta=5000000,
            rope_scaling={"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
            tie_word_embeddings=False,
        ),
        vision_config=dict(
            depth=3,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            num_position_embeddings=64,
            # Two DeepStack levels, added after decoder layers 0 and 1, so the
            # tap at LAYER sits past every add as the real one (layer 50,
            # three levels) does.
            deepstack_visual_indexes=[0, 2],
            hidden_act="gelu_pytorch_tanh",
        ),
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)

    model = Qwen3VLForConditionalGeneration(tiny_config()).eval().to(torch.float32)
    # Random init leaves the norms at one and the biases at zero; perturb
    # every parameter so a dropped norm or bias cannot hide.
    for p in model.parameters():
        p.add_(0.02 * torch.randn_like(p))
    enc_dir = os.path.join(args.out, "text_encoder")
    model.save_pretrained(enc_dir, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(os.path.join(args.h3_dir, "processor"))
    tokenizer = processor.tokenizer

    image = rng.integers(0, 256, (96, 160, 3), dtype=np.uint8)
    video = rng.integers(0, 256, (30, 64, 96, 3), dtype=np.uint8)
    audio = torch.zeros(2, 3200)
    references = [
        MiniMaxH3ImageReference(image=Image.fromarray(image)),
        MiniMaxH3AudioReference(audio=audio, sample_rate=32000),
        MiniMaxH3VideoReference(frames=video, fps=FPS, audio=audio, sample_rate=32000),
    ]

    step = MiniMaxH3Ref2VATextEncoderStep()
    vision_inputs, image_counts, video_counts, video_timestamps = step._gather_vision_features(processor, references, FPS)
    token_ids, token_tags = step._build_presentation(
        tokenizer, PROMPT, references, image_counts, video_counts, video_timestamps,
        text_tag=MINIMAX_H3_TEXT_TAG, video_tag=MINIMAX_H3_VIDEO_TAG,
    )
    hidden = get_qwen3vl_prompt_embeds(model, processor, token_ids, vision_inputs, text_encoder_layer=LAYER, dtype=torch.float32)[0]

    # The vision tower's own outputs, per modality: the merged visual rows
    # and every DeepStack level, so a vision-side drift is told apart from a
    # language-side one.
    taps = {}
    for kind, pixels, grid in (("image", "pixel_values", "image_grid_thw"), ("video", "pixel_values_videos", "video_grid_thw")):
        out = model.model.visual(vision_inputs[pixels].to(torch.float32), grid_thw=vision_inputs[grid], return_dict=True)
        taps[f"{kind}_visual"] = out.pooler_output.contiguous()
        for level, feat in enumerate(out.deepstack_features):
            taps[f"{kind}_deepstack_{level}"] = feat.contiguous()

    tensors = {
        **taps,
        "image": torch.from_numpy(image.astype(np.float32)),
        "video": torch.from_numpy(video.astype(np.float32)),
        "token_ids": torch.tensor(token_ids, dtype=torch.float32),
        "token_tags": torch.tensor(token_tags, dtype=torch.float32),
        "hidden": hidden.contiguous(),
    }
    path = os.path.join(args.out, "minimaxh3_presentation.safetensors")
    save_file(tensors, path)
    manifest = {
        "minimaxh3_presentation.safetensors": {
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "tensors": {k: list(v.shape) for k, v in tensors.items()},
        },
        "prompt": PROMPT,
        "seed": args.seed,
        "video_block_timestamps": video_timestamps,
        "source": source_block(
            checkpoint="MiniMaxAI/MiniMax-H3 (processor + tokenizer only; tiny random Qwen3-VL)",
            files=[os.path.join(args.h3_dir, "processor", "preprocessor_config.json"), os.path.join(args.h3_dir, "processor", "video_preprocessor_config.json")],
            identity={"layer": LAYER, "fps_x1000": int(FPS * 1000), "hidden": 64},
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print("wrote", {k: list(v.shape) for k, v in tensors.items()}, "timestamps", video_timestamps)


if __name__ == "__main__":
    main()
