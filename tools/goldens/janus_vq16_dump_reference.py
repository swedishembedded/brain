#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reference-parity image tokenizers for its
# clients. If your team needs a VQ image tokenizer ported and proven against
# its reference, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Dump Janus-Pro / LlamaGen VQ-16 reference goldens for `crates/vqgan`.

Janus-Pro's image tokenizer (`gen_vision_model`) is LlamaGen's `VQ_16`,
vendored verbatim as `janus/models/vq_model.py`. This script imports THAT file
directly (not the `janus` package, whose `__init__` pulls in transformers),
builds `VQ_16()` with the defaults Janus itself constructs it with
(`modeling_vlm.py`: `gen_vision_cls()`), loads every `gen_vision_model.*`
tensor from the Janus-Pro shard that holds it, and runs it on the CPU in fp32.

It writes, to `<out>/` (all floats f32, indices I32):

  decode.safetensors   `codes` I32 [2, 576]: image 0 is the encode indices of
                       the synthetic image below (a realistic code occupancy),
                       image 1 is seeded uniform codes over the whole codebook.
                       `pixels` [2, 3, 384, 384] = `decode_code(codes,
                       shape=[2, 8, 24, 24])`, plus stage taps of image 0
                       named by their reference module path (`decoder.mid.2`,
                       `decoder.conv_blocks.1.upsample`, ...).
  encode.safetensors   `image` [3, 384, 384] synthetic input in [-1, 1],
                       `indices` I32 [576] from `encode(image)`, the
                       quantizer query (the `quant_conv` tap, [8, 24, 24]),
                       its L2-normalised rows `z_norm` [576, 8], the
                       per-query distance margin `margin` [576] (second-best
                       minus best distance, the near-tie diagnostic), and
                       encoder stage taps by module path.
  manifest.json        per-file sha256, tensor shapes, the `source` block
                       (`golden_source.py`), run params and versions.

The reference venv's torch (2.0.1) cannot exchange tensors with numpy, so
nothing here touches numpy; `safetensors.torch.save_file` writes directly.

usage:
  python tools/goldens/janus_vq16_dump_reference.py \
      --janus   /path/to/Janus \
      --weights /path/to/deepseek-ai/Janus-Pro-7B \
      --out     /tmp/brain-testdata/janus/vq16
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

PREFIX = "gen_vision_model."
IMG = 384          # Janus-Pro generation resolution (generation_inference.py)
GRID = IMG // 16   # VQ-16 downsamples by 16: 24x24 = 576 codes per image


def load_vq_model(janus_dir):
    path = os.path.join(janus_dir, "janus", "models", "vq_model.py")
    spec = importlib.util.spec_from_file_location("janus_vq_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gen_vision_state(weights_dir):
    """Every `gen_vision_model.*` tensor, prefix stripped, from the shard(s)
    the index names for it. Returns (state_dict, shard paths read)."""
    with open(os.path.join(weights_dir, "pytorch_model.bin.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    shards = sorted({v for k, v in wmap.items() if k.startswith(PREFIX)})
    assert shards, f"no {PREFIX}* tensors in the index"
    sd, dtypes = {}, set()
    for s in shards:
        path = os.path.join(weights_dir, s)
        print(f"  reading {s} ...", flush=True)
        full = torch.load(path, map_location="cpu", weights_only=True)
        for k, v in full.items():
            if k.startswith(PREFIX):
                dtypes.add(str(v.dtype))
                sd[k[len(PREFIX):]] = v.float()
        del full
    want = {k[len(PREFIX):] for k in wmap if k.startswith(PREFIX)}
    assert set(sd) == want, f"index/shard mismatch: {sorted(want ^ set(sd))[:8]}"
    return sd, [os.path.join(weights_dir, s) for s in shards], sorted(dtypes)


def det_image(h, w, seed):
    """Deterministic RGB pattern in [-1, 1], shape (1, 3, h, w): smooth
    structure plus seeded noise, so the code occupancy is neither trivial nor
    pure noise."""
    ys = torch.linspace(0, 3.14159, h).unsqueeze(1).expand(h, w)
    xs = torch.linspace(0, 6.28318, w).unsqueeze(0).expand(h, w)
    r = torch.sin(3.0 * xs + ys)
    g = torch.cos(2.0 * xs) * torch.sin(0.5 * ys)
    b = 2.0 * (ys / 3.14159) - 1.0
    img = torch.stack([r, g, b], 0)
    gen = torch.Generator().manual_seed(seed)
    img = img + 0.15 * torch.randn(img.shape, generator=gen)
    return img.clamp(-1.0, 1.0).unsqueeze(0).contiguous()


class Taps:
    """Forward hooks; keeps batch item 0 of each output, named by path."""

    def __init__(self):
        self.t, self.h = {}, []

    def on(self, model, path):
        mod = model.get_submodule(path)

        def f(_m, _i, out):
            self.t[path] = out[0].detach().float().clone().contiguous()
        self.h.append(mod.register_forward_hook(f))

    def off(self):
        for h in self.h:
            h.remove()
        self.h = []


ENC_TAPS = (["encoder.conv_in"]
            + [f"encoder.conv_blocks.{i}.downsample" for i in range(4)]
            + ["encoder.conv_blocks.4.attn.1", "encoder.mid.0", "encoder.mid.1",
               "encoder.mid.2", "encoder.conv_out", "quant_conv"])
# Full-resolution (384^2) decoder taps are left out to keep the golden small;
# the last upsample (192 -> 384) and conv_out bracket that level.
DEC_TAPS = (["post_quant_conv", "decoder.conv_in", "decoder.mid.0", "decoder.mid.1",
             "decoder.mid.2", "decoder.conv_blocks.0.attn.2"]
            + [f"decoder.conv_blocks.{i}.upsample" for i in range(3)]
            + ["decoder.conv_out"])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def save(out_dir, name, tensors, manifest):
    tensors = {k: (v if v.dtype == torch.int32 else v.to(torch.float32)).contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out_dir, name)
    save_file(tensors, path, metadata={"src": "Janus vq_model.VQ_16 fp32 CPU"})
    manifest["files"][name] = {
        "sha256": sha256_file(path),
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                    for k, v in sorted(tensors.items())},
    }
    print(f"  wrote {name} ({len(tensors)} tensors, "
          f"{os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--janus", required=True, help="Janus repo root")
    ap.add_argument("--weights", required=True, help="Janus-Pro-7B checkpoint dir")
    ap.add_argument("--out", required=True, help="golden output dir")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    vq = load_vq_model(args.janus)
    model = vq.VQ_16()
    sd, shard_paths, dtypes = gen_vision_state(args.weights)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model = model.float().eval()
    cfg = model.config
    print(f"  loaded {len(sd)} tensors (stored as {dtypes})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "reference": {
            "module": "janus/models/vq_model.py VQ_16() (LlamaGen VQ-16)",
            "device": "cpu", "dtype": "float32",
        },
        "run": {"seed": args.seed, "image": IMG, "grid": GRID,
                "weights_dtype": dtypes, "tensors": len(sd)},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "source": source_block(
            checkpoint="deepseek-ai/Janus-Pro-7B",
            files=shard_paths,
            identity={
                "codebook_size": cfg.codebook_size,
                "codebook_embed_dim": cfg.codebook_embed_dim,
                "z_channels": cfg.z_channels,
                "levels": len(cfg.decoder_ch_mult),
            },
            # A 10 GB shard; the identity carries the enforced half.
            hash_files=False,
        ),
        "files": {},
    }

    # ---- encode ------------------------------------------------------------
    x = det_image(IMG, IMG, args.seed)
    taps = Taps()
    for p in ENC_TAPS:
        taps.on(model, p)
    quant, _, (_, _, idx) = model.encode(x)
    taps.off()
    idx = idx.reshape(-1).to(torch.int32)
    assert idx.numel() == GRID * GRID

    # Recompute the assignment exactly as VectorQuantizer.forward does, and
    # record each query's margin to its runner-up code.
    z = taps.t["quant_conv"]                                   # (8, 24, 24)
    zf = torch.nn.functional.normalize(z.permute(1, 2, 0).reshape(-1, cfg.codebook_embed_dim), p=2, dim=-1)
    emb = torch.nn.functional.normalize(model.quantize.embedding.weight, p=2, dim=-1)
    d = (zf ** 2).sum(1, keepdim=True) + (emb ** 2).sum(1) - 2 * zf @ emb.t()
    assert torch.equal(d.argmin(1).to(torch.int32), idx), "recomputed argmin != reference"
    top2 = d.topk(2, dim=1, largest=False).values
    margin = top2[:, 1] - top2[:, 0]
    print(f"  encode: {idx.unique().numel()} distinct codes, min margin {margin.min():.3e}, "
          f"{int((margin < 1e-6).sum())} queries under 1e-6", flush=True)

    enc = dict(taps.t)
    enc["image"] = x[0]
    enc["indices"] = idx
    enc["z_norm"] = zf
    enc["margin"] = margin
    enc["quant"] = quant[0]
    save(args.out, "encode.safetensors", enc, manifest)

    # ---- decode ------------------------------------------------------------
    g = torch.Generator().manual_seed(args.seed + 1)
    rnd = torch.randint(0, cfg.codebook_size, (GRID * GRID,), generator=g, dtype=torch.int64)
    codes = torch.stack([idx.to(torch.int64), rnd], 0)          # (2, 576)
    taps = Taps()
    for p in DEC_TAPS:
        taps.on(model, p)
    pixels = model.decode_code(codes, shape=[2, cfg.codebook_embed_dim, GRID, GRID])
    taps.off()
    print(f"  decode: pixels {list(pixels.shape)}, range [{pixels.min():.3f}, {pixels.max():.3f}]",
          flush=True)
    dec = dict(taps.t)
    dec["codes"] = codes.to(torch.int32)
    dec["pixels"] = pixels
    save(args.out, "decode.safetensors", dec, manifest)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"  wrote manifest.json -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
