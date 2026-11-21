#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the reference dequantization of NVFP4 and MXFP8 expert slabs of the
published Inkling 2 Mini checkpoint, in f32.

The values come from the reference loader's own op
(`transformers.core_model_loading.Inkling2LoopDequantizeExperts`, at the pinned
transformers commit), asked for float32 output instead of the bf16 it casts to
by default, so a brain decode can be compared bit for bit: NVFP4 is
`e2m1 * e4m3 block scale * (amax / outer_scale_denominator)`, MXFP8 is
`e4m3 * 2^(e8m0 - 127)`.

Usage (the checkpoint directory comes from the model store):

  python3 -I tools/goldens/inkling2_expert_dequant_reference.py --model-dir <store-dir>

Writes to `--out` (default `testdata/golden/inkling2/experts/`):
  experts.safetensors  `<format>.<tensor>.<expert>`: each the [rows, cols] f32 slab
  manifest.json        which tensors and experts, the source block, versions
"""

import argparse
import json
import os
import sys

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.core_model_loading import Inkling2LoopDequantizeExperts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

# (format, stored tensor, expert): layers 0-19 are NVFP4, 20-23 MXFP8 (per hf_quant_config.json).
SLABS = [
    ("nvfp4", "model.llm.layers.0.mlp.experts.w13_weight", 5),
    ("nvfp4", "model.llm.layers.0.mlp.experts.w2_weight", 127),
    ("mxfp8", "model.llm.layers.20.mlp.experts.w13_weight", 3),
    ("mxfp8", "model.llm.layers.20.mlp.experts.w2_weight", 64),
]


class _F32Target:
    """What the op asks of a model: the dtype of the parameter it fills."""

    def get_parameter(self, _name):
        return torch.empty(0, dtype=torch.float32)


def dequant(f, name, expert):
    patterns = [name, name + "_scale", name + "_amax", name + "_outer_scale_denominator"]
    present = set(f.keys())
    inputs = {}
    for p in patterns:
        if p not in present:
            continue
        # One expert's slab of each stacked tensor; the denominator is one value for every expert.
        inputs[p] = f.get_tensor(p) if p.endswith("_outer_scale_denominator") else f.get_slice(p)[expert : expert + 1]
    out = Inkling2LoopDequantizeExperts().convert(inputs, patterns, ["w"], model=_F32Target(), full_layer_name="w")["w"]
    assert out.dtype == torch.float32 and out.shape[0] == 1, (out.dtype, out.shape)
    return out[0].contiguous()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="testdata/golden/inkling2/experts")
    args = ap.parse_args()
    weights = os.path.join(args.model_dir, "model-00001-of-00001.safetensors")
    tensors = {}
    with safe_open(weights, "pt") as f:
        for fmt, name, expert in SLABS:
            tensors[f"{fmt}.{name}.{expert}"] = dequant(f, name, expert)
    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "experts.safetensors"))
    with open(os.path.join(args.model_dir, "config.json")) as c:
        text = json.load(c)["text_config"]
    manifest = {
        "slabs": [{"format": fmt, "tensor": name, "expert": expert} for fmt, name, expert in SLABS],
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "source": source_block(
            files=[weights],
            hash_files=False,
            identity={"hidden_size": text["hidden_size"], "intermediate_size": text["intermediate_size"], "n_routed_experts": text["n_routed_experts"]},
        ),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as m:
        json.dump(manifest, m, indent=2)


if __name__ == "__main__":
    main()
