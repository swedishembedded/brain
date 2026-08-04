#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the FastVLM-0.5B mobileclip vision-tower features (and the preprocessed
pixels) so brain's own FastViTHD (`Encoder::mobileclip_l`) can be parity-checked on
the real weights — closing the loop for a fully-in-brain image caption.

Outputs (little-endian f32):
  parity/fastvlm_vis_px.bin    [1, 3, 1024, 1024] preprocessed pixels
  parity/fastvlm_vis_feat.bin  [256, 3072] vision-tower features (pre-projector)
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

CKPT = os.environ.get("BRAIN_FASTVLM_CKPT", "/data/workspace/resources/vl/fastvlm/hf/FastVLM-0.5B")
IMG = os.environ.get("BRAIN_FASTVLM_TEST_IMG", "/data/workspace/resources/emulation/dosbox-x/DOSBox-Logo-2-680x350.png")
OUT = os.environ.get("BRAIN_VL_PARITY_OUT", "/data/workspace/resources/vl/parity")
os.makedirs(OUT, exist_ok=True)

m = AutoModelForCausalLM.from_pretrained(CKPT, trust_remote_code=True, dtype=torch.float32, low_cpu_mem_usage=True).eval()
vt = m.get_vision_tower()
px = vt.image_processor(Image.open(IMG).convert("RGB"), return_tensors="pt")["pixel_values"].float()
with torch.no_grad():
    feat = vt(px)  # [1, 256, 3072]

px.numpy().astype("<f4").tofile(f"{OUT}/fastvlm_vis_px.bin")
feat[0].numpy().astype("<f4").tofile(f"{OUT}/fastvlm_vis_feat.bin")
print(f"pixels {tuple(px.shape)}, vision features {tuple(feat.shape)}")
