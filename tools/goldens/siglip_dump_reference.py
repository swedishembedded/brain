#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements vision-language model ports with proven
# reference parity for its clients. If your team needs expertise in porting
# vision encoders to custom inference engines then you can procure our
# services by sending an email to info@swedishembedded.com.

"""Dump SigLIP-L/16@384 reference features for `crates/clip`'s SigLIP stem.

The tower is `siglip_large_patch16_384` exactly as DeepSeek-VL builds it
(`deepseek_vl.models.siglip_vit.create_siglip_vit`, a pinned copy of timm's
`VisionTransformer`: no class token, a biased patch conv, a per-patch learned
position table, pre-LN blocks with exact-erf GELU, LayerNorm eps 1e-6 and a
final `norm`). Janus-Pro's understanding tower is the same builder and the same
config (`select_layer=-1`, `select_feature="same"`), so one golden covers both
checkpoints' math.

Weights: DeepSeek-VL-7B-chat's low-resolution tower,
`vision_model.vision_tower_low.vision_tower.*`, fp16 in the checkpoint and cast
to fp32 here - the same cast brain's importer applies. The `attn_pool.*`
tensors load (the reference module builds the MAP head because
`global_pool="map"`) but never run: `ignore_head=True`, so `forward` stops at
`forward_features`.

Input: a fixed synthetic `[1, 3, 384, 384]` image already in normalized pixel
space (the reference normalizes with mean = std = 0.5, i.e. values in
[-1, 1]); it is saved so the Rust side replays it rather than regenerating it.

Outputs, little-endian f32, under `<testdata>/siglip/deepseek_vl_low/`:
  pixels.bin        [1, 3, 384, 384]
  block_00.bin      [576, 1024]  output of blocks[0]
  block_11.bin      [576, 1024]  output of blocks[11]
  block_23.bin      [576, 1024]  output of blocks[23] (pre final norm)
  features.bin      [576, 1024]  forward() output (post final norm)
  manifest.json     shapes, sha256, run parameters and the `source` block

Usage (the venv needs torch, timm and safetensors; its torch cannot convert
to numpy, so every tensor leaves through `.tolist()`):
  python tools/goldens/siglip_dump_reference.py \
      --reference <DeepSeek-VL repo checkout> [--models-dir DIR] [--testdata DIR]
"""

import argparse
import array
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402  (tools/goldens is this file's own dir)

CHECKPOINT = "deepseek-ai/deepseek-vl-7b-chat"
PREFIX = "vision_model.vision_tower_low.vision_tower."
MODEL_NAME = "siglip_large_patch16_384"
IMAGE_SIZE = 384
TAPS = (0, 11, 23)


def default_models_dir():
    if os.environ.get("BRAIN_MODELS_DIR"):
        return os.environ["BRAIN_MODELS_DIR"]
    if os.environ.get("XDG_DATA_HOME"):
        return os.path.join(os.environ["XDG_DATA_HOME"], "brain", "models")
    return os.path.join(os.path.expanduser("~"), ".local", "share", "brain", "models")


def default_testdata():
    return os.environ.get("BRAIN_TESTDATA") or "testdata"


def synthetic_pixels(torch):
    """A smooth, channel-distinct pattern with edges in it, in [-1, 1].

    Deterministic by construction (no RNG), so the golden is reproducible
    without pinning a generator."""
    s = IMAGE_SIZE
    y = torch.arange(s, dtype=torch.float32).view(s, 1).expand(s, s)
    x = torch.arange(s, dtype=torch.float32).view(1, s).expand(s, s)
    chans = [
        torch.sin(x * 0.031 + y * 0.017),
        torch.cos(x * 0.011 - y * 0.029) * 0.8,
        torch.where((x.long() // 48 + y.long() // 48) % 2 == 0, 0.6, -0.6) + 0.3 * torch.sin((x + y) * 0.05),
    ]
    return torch.stack(chans).clamp(-1.0, 1.0).unsqueeze(0).contiguous()


def write_f32(path, t):
    """`t` as raw little-endian f32, via `.tolist()` (no numpy bridge)."""
    flat = array.array("f", t.detach().to("cpu").float().reshape(-1).tolist())
    if sys.byteorder != "little":
        flat.byteswap()
    with open(path, "wb") as f:
        flat.tofile(f)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return {"shape": list(t.shape), "dtype": "f32", "sha256": h.hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="DeepSeek-VL repository checkout (provides deepseek_vl.models.siglip_vit)")
    ap.add_argument("--models-dir", default=default_models_dir())
    ap.add_argument("--testdata", default=default_testdata())
    args = ap.parse_args()

    import torch
    from safetensors import safe_open

    sys.path.insert(0, args.reference)
    from deepseek_vl.models.siglip_vit import create_siglip_vit

    ckpt = os.path.join(args.models_dir, *CHECKPOINT.split("/"))
    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({f for k, f in index.items() if k.startswith(PREFIX)})
    if not shards:
        raise SystemExit(f"{ckpt}: no tensor under {PREFIX}")
    state = {}
    for shard in shards:
        with safe_open(os.path.join(ckpt, shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith(PREFIX):
                    state[k[len(PREFIX):]] = f.get_tensor(k).float()

    torch.manual_seed(0)
    model = create_siglip_vit(MODEL_NAME, IMAGE_SIZE, select_layer=-1).float().eval()
    # strict: every checkpoint tensor has a home in the module and vice versa.
    model.load_state_dict(state, strict=True)
    blocks = len(model.blocks)
    if blocks != 24:
        raise SystemExit(f"expected 24 blocks at select_layer=-1, built {blocks}")

    taps = {}
    for i in TAPS:
        model.blocks[i].register_forward_hook(lambda _m, _inp, out, i=i: taps.__setitem__(i, out.detach().clone()))

    px = synthetic_pixels(torch)
    with torch.no_grad():
        feats = model(px)
    if tuple(feats.shape) != (1, 576, 1024):
        raise SystemExit(f"features shape {tuple(feats.shape)}, expected (1, 576, 1024)")
    # Self-check: the final output is the final norm of the last block's output.
    with torch.no_grad():
        renorm = model.norm(taps[23])
    drift = (renorm - feats).abs().max().item()
    if drift != 0.0:
        raise SystemExit(f"forward() is not norm(blocks[23]) (max |d| {drift}) - the tap is misplaced")

    out = os.path.join(args.testdata, "siglip", "deepseek_vl_low")
    os.makedirs(out, exist_ok=True)
    files = {"pixels.bin": write_f32(os.path.join(out, "pixels.bin"), px)}
    for i in TAPS:
        files[f"block_{i:02d}.bin"] = write_f32(os.path.join(out, f"block_{i:02d}.bin"), taps[i][0])
    files["features.bin"] = write_f32(os.path.join(out, "features.bin"), feats[0])

    manifest = {
        "model": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "taps": list(TAPS),
        "files": files,
        "versions": {"torch": torch.__version__},
        "source": source_block(
            checkpoint=CHECKPOINT,
            files=[os.path.join(ckpt, s) for s in shards],
            identity={"width": 1024, "layers": blocks, "heads": 16, "patch": 16, "image_size": IMAGE_SIZE},
        ),
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}: features {tuple(feats.shape)}, taps {list(TAPS)}")


if __name__ == "__main__":
    main()
