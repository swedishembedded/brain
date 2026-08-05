#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump CodeFormer *restoration* reference goldens for brain's `crates/restore`.

`tools/codeformer_dump_reference.py` already gates the VQGAN core (encoder /
codebook / generator, `crates/vqgan`). This script gates what CodeFormer adds on
top of it and what `crates/restore` implements:

  * the **code-prediction Transformer** — `feat_emb` -> 9 x `TransformerSALayer`
    (pre-LN, 8-head bidirectional self-attention with a learned position
    embedding added to q and k but NOT to v, erf-GELU MLP) -> `idx_pred_layer`
    (LayerNorm + biasless Linear) -> 1024-way code logits, argmax -> indices;
  * the **controllable feature transformation** (`Fuse_sft_block`) between the
    encoder features and the generator;
  * the **identity-fidelity dial `w`** — the reference scales the CFT residual
    by `w` and skips the block entirely at `w == 0`, so w=0 is maximum quality
    (pure code-predicted reconstruction) and w=1 maximum fidelity to the input.
    The direction is invisible in a single-w dump, so every w is dumped and the
    dumper asserts the endpoints differ in the expected direction.

Everything is captured with **forward hooks during one real `model(x, w=...)`
call**, CPU + fp32 + fixed seed. Stage boundaries are where a port breaks.

Written per variant (`codeformer`), all tensors f32 except indices (I32):

  encoder.safetensors   the four CFT encoder taps (blocks 5/8/11/14) + `lq_feat`
                        (block 24). w-independent, so dumped once.
  transformer.safetensors  `feat_emb` output, every `ft_layers.{i}` output,
                        the `idx_pred_layer` logits, the argmax indices, the
                        gathered `quant_feat`, plus INSIDE layer 0: norm1, the
                        attention output, the post-attention residual, norm2 and
                        the MLP output — full bisection of one layer.
  gen_w{tag}.safetensors  per w: every generator block that feeds or follows a
                        fuse point (pre-fuse), the post-fuse tensor, the
                        `Fuse_sft_block` internals (encode_enc / scale / shift)
                        and the final image.
  manifest.json         per-file sha256 + every tensor's shape and dtype, the
                        w grid, run params and library versions.

usage:
  python3 tools/goldens/codeformer_restore_dump_reference.py \
      --code    /path/to/CodeFormer \
      --weights /path/to/weights/codeformer \
      --out     testdata/restore/codeformer \
      --face    /path/to/aligned_face_512.png
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types

import torch
from safetensors.torch import save_file

# CodeFormer(dim_embd, n_head, n_layers, codebook_size, latent_size,
#            connect_list) exactly as `inference_codeformer.py` constructs it.
CF_CONFIG = dict(dim_embd=512, n_head=8, n_layers=9, codebook_size=1024,
                 latent_size=256, connect_list=['32', '64', '128', '256'])

# codeformer_arch.py:204-206 — the taps the CFT connects, for `connect_list`.
FUSE_ENC = {'256': 5, '128': 8, '64': 11, '32': 14}
FUSE_GEN = {'32': 9, '64': 12, '128': 15, '256': 18}
LQ_BLOCK = 24            # encoder.blocks[24] == lq_feat
GEN_LAST = 24            # generator.blocks[24] == the image head


def _stub_basicsr():
    """Register the two `basicsr` names the arch files import.

    `import basicsr` pulls in losses -> lpips, which is not installed and is
    irrelevant here.
    """
    utils = types.ModuleType("basicsr.utils")
    utils.get_root_logger = lambda *a, **k: __import__("logging").getLogger("codeformer")
    registry = types.ModuleType("basicsr.utils.registry")

    class _Reg:
        def register(self, cls=None, **k):
            return cls if cls is not None else (lambda c: c)
    registry.ARCH_REGISTRY = _Reg()
    utils.registry = registry
    pkg = types.ModuleType("basicsr")
    pkg.utils = utils
    archs = types.ModuleType("basicsr.archs")
    pkg.archs = archs
    for name, mod in [("basicsr", pkg), ("basicsr.utils", utils),
                      ("basicsr.archs", archs),
                      ("basicsr.utils.registry", registry)]:
        sys.modules[name] = mod


def _exec(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_arch(code_dir):
    """Import CodeFormer's `codeformer_arch.py` (and the vqgan_arch it star-imports)."""
    _stub_basicsr()
    base = os.path.join(code_dir, "basicsr", "archs")
    vq = _exec(os.path.join(base, "vqgan_arch.py"), "basicsr.archs.vqgan_arch")
    sys.modules["vqgan_arch"] = vq
    return _exec(os.path.join(base, "codeformer_arch.py"), "codeformer_arch")


def build(arch, ckpt_path):
    """Build CodeFormer and load `params_ema` with STRICT coverage (never zero-fill)."""
    m = arch.CodeFormer(**CF_CONFIG)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "params_ema" in ck, f"{ckpt_path}: expected 'params_ema', got {list(ck)}"
    sd = ck["params_ema"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, f"missing tensors: {missing[:8]}"
    assert not unexpected, f"unexpected tensors: {unexpected[:8]}"
    return m.float().eval(), len(sd)


def det_image(h, w, seed):
    """Deterministic RGB test pattern in [-1, 1], shape (1, 3, h, w).

    The same pattern `tools/codeformer_dump_reference.py` uses, so the VQGAN
    goldens and these share an input on the synthetic case.
    """
    ys = torch.linspace(0, 3.14159, h).unsqueeze(1).expand(h, w)
    xs = torch.linspace(0, 6.28318, w).unsqueeze(0).expand(h, w)
    r = torch.sin(3.0 * xs + ys)
    g = torch.cos(2.0 * xs) * torch.sin(0.5 * ys)
    b = 2.0 * (ys / 3.14159) - 1.0
    img = torch.stack([r, g, b], 0)
    gen = torch.Generator().manual_seed(seed)
    img = img + 0.15 * torch.randn(img.shape, generator=gen)
    return img.clamp(-1.0, 1.0).unsqueeze(0).contiguous()


def face_image(path):
    """A real 512x512 aligned face -> (1, 3, 512, 512) in [-1, 1]."""
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("RGB")
    assert im.size == (512, 512), f"{path}: expected 512x512, got {im.size}"
    a = torch.from_numpy(np.asarray(im).copy()).float() / 255.0     # (H, W, 3)
    return (a.permute(2, 0, 1) * 2.0 - 1.0).unsqueeze(0).contiguous()


class Taps:
    """Forward-hook capture; the leading batch axis is stripped (every run is b=1)."""

    def __init__(self):
        self.t, self.h = {}, []

    def on(self, module, name, pick=lambda o: o, strip_batch=True):
        def f(_m, _i, out):
            v = pick(out)
            v = v[0] if strip_batch else v
            self.t[name] = v.detach().float().clone().contiguous()
        self.h.append(module.register_forward_hook(f))

    def pre(self, module, name, pick=lambda i: i[0]):
        def f(_m, inp):
            self.t[name] = pick(inp).detach().float().clone().contiguous()
        self.h.append(module.register_forward_pre_hook(f))

    def off(self):
        for h in self.h:
            h.remove()
        self.h = []


def tap_transformer(model, taps):
    """Encoder taps + the whole code-prediction transformer, layer 0 in detail.

    Transformer activations are (HW, B, C) with B = 1; the batch axis is axis 1,
    so `strip_batch` (axis 0) is wrong for them — they are squeezed explicitly
    into the [T, C] rows brain compares.
    """
    for size, i in FUSE_ENC.items():
        taps.on(model.encoder.blocks[i], f"enc.{i:02d}")
    taps.on(model.encoder.blocks[LQ_BLOCK], "lq_feat")

    def seq(o):
        return o.squeeze(1) if o.dim() == 3 else o

    taps.on(model.feat_emb, "feat_emb", pick=seq, strip_batch=False)
    for i, layer in enumerate(model.ft_layers):
        taps.on(layer, f"ft.{i:02d}", pick=seq, strip_batch=False)
    # Inside layer 0: the five stage boundaries of one TransformerSALayer.
    l0 = model.ft_layers[0]
    taps.on(l0.norm1, "ft.00.norm1", pick=seq, strip_batch=False)
    taps.on(l0.self_attn, "ft.00.attn_out", pick=lambda o: seq(o[0]), strip_batch=False)
    taps.on(l0.norm2, "ft.00.norm2", pick=seq, strip_batch=False)
    taps.on(l0.linear1, "ft.00.linear1", pick=seq, strip_batch=False)
    taps.on(l0.linear2, "ft.00.linear2", pick=seq, strip_batch=False)
    # idx_pred_layer = Sequential(LayerNorm, Linear) -> (HW, B, N)
    taps.on(model.idx_pred_layer[0], "logits_norm", pick=seq, strip_batch=False)
    taps.on(model.idx_pred_layer, "logits", pick=seq, strip_batch=False)


def tap_generator(model, taps):
    """Every generator block at or around a fuse point, plus the fuse internals.

    `gen.{i}` is the block output BEFORE the fuse; `fuse.{size}.out` is the same
    position AFTER it. At w=0 the reference skips the fuse entirely, so the two
    are identical there by construction — which is exactly what the port's
    `w = 0` case must reproduce.
    """
    for i in sorted(set(list(FUSE_GEN.values()) + [GEN_LAST])):
        taps.on(model.generator.blocks[i], f"gen.{i:02d}")
    for size, blk in model.fuse_convs_dict.items():
        taps.on(blk, f"fuse.{size}.out")
        taps.on(blk.encode_enc, f"fuse.{size}.encode_enc")
        taps.on(blk.scale, f"fuse.{size}.scale")
        taps.on(blk.shift, f"fuse.{size}.shift")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(out_dir, name, tensors, manifest):
    tensors = {k: (v if v.dtype == torch.int32 else v.to(torch.float32)).contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out_dir, name)
    save_file(tensors, path, metadata={"src": "CodeFormer codeformer_arch fp32 CPU"})
    manifest[name] = {
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                    for k, v in sorted(tensors.items())},
    }
    print(f"  wrote {name}  ({len(tensors)} tensors, "
          f"{os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def wtag(w):
    return f"{w:.2f}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="CodeFormer repo root")
    ap.add_argument("--weights", required=True, help="dir holding codeformer.pth")
    ap.add_argument("--out", required=True, help="golden output root")
    ap.add_argument("--face", default=None, help="512x512 aligned face png")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ws", default="0.0,0.25,0.5,0.75,1.0")
    args = ap.parse_args()

    arch = load_arch(args.code)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    ws = [float(v) for v in args.ws.split(",")]

    ckpt = os.path.join(args.weights, "codeformer.pth")
    model, n_tensors = build(arch, ckpt)
    print(f"=== codeformer ({ckpt}): {n_tensors} tensors loaded strict", flush=True)

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "config": CF_CONFIG,
        "seed": args.seed,
        "w_grid": ws,
        "checkpoint": {"path": os.path.abspath(ckpt), "sha256": sha256_file(ckpt),
                       "key": "params_ema", "tensors_total": n_tensors},
        "fuse_taps": {"encoder": FUSE_ENC, "generator": FUSE_GEN},
        "reference": {"arch": "basicsr.archs.codeformer_arch.CodeFormer",
                      "code": os.path.abspath(args.code),
                      "device": "cpu", "dtype": "float32"},
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "notes": [
            "Transformer activations are (HW, B, C) in the reference; every golden "
            "here is squeezed to [T, C] with T = HW = 256, C = 512.",
            "TransformerSALayer is PRE-norm: tgt2 = norm1(tgt); q = k = tgt2 + "
            "position_emb; v = tgt2 (the position embedding is NOT added to v); "
            "tgt = tgt + attn(q,k,v); tgt = tgt + linear2(gelu(linear1(norm2(tgt)))).",
            "nn.MultiheadAttention: 8 heads, head_dim 64, q scaled by 1/sqrt(64), "
            "softmax over the full key axis (bidirectional, no mask).",
            "F.gelu default is the ERF form, not the tanh approximation.",
            "idx_pred_layer = Sequential(LayerNorm(512), Linear(512, 1024, bias=False)); "
            "argmax over the 1024 logits == topk(softmax(logits), 1) since softmax is "
            "monotone. Ties -> lowest index.",
            "quant_feat = quantize.get_codebook_feat(top_idx, [b,16,16,256]) — the RAW "
            "codebook gather, NOT the straight-through form.",
            "Fuse_sft_block: e = encode_enc(cat([enc_feat, dec_feat], dim=1)); "
            "out = dec_feat + w * (dec_feat * scale(e) + shift(e)). scale/shift are "
            "Conv3x3 -> LeakyReLU(0.2) -> Conv3x3.",
            "w == 0: the reference SKIPS the fuse block entirely, which is bit-identical "
            "to evaluating it and multiplying the residual by 0. w = 0 is maximum "
            "QUALITY (generator runs on the predicted codes alone); w = 1 is maximum "
            "FIDELITY (full encoder-feature injection).",
        ],
    }

    cases = [("synth", det_image(512, 512, args.seed))]
    if args.face:
        cases.append(("face", face_image(args.face)))
        manifest["face_image"] = os.path.abspath(args.face)

    for tag, x in cases:
        print(f"  --- case '{tag}'", flush=True)
        # ---- pass 1: encoder + transformer (w-independent) -----------------
        taps = Taps()
        tap_transformer(model, taps)
        out0, logits, lq = model(x, w=0.0, detach_16=True, adain=False)
        taps.off()
        t = dict(taps.t)
        t["input"] = x[0].contiguous()
        idx = logits.softmax(dim=2).topk(1, dim=2)[1].reshape(-1)
        t["indices"] = idx.to(torch.int32)
        quant = model.quantize.get_codebook_feat(idx.view(-1, 1), shape=[1, 16, 16, 256])
        t["quant_feat"] = quant[0].contiguous()
        # Self-validation: argmax(logits) must equal topk(softmax(logits)).
        assert torch.equal(logits[0].argmax(dim=1).to(torch.int32), t["indices"]), \
            "argmax(logits) != topk(softmax(logits)) — a tie flipped"
        n_unique = int(idx.unique().numel())
        print(f"    codes predicted: {n_unique}/1024 distinct over {idx.numel()} positions",
              flush=True)
        t["n_unique_codes"] = torch.tensor([float(n_unique)])
        save(args.out, f"encoder_{tag}.safetensors",
             {k: v for k, v in t.items()
              if k in ("input", "lq_feat") or k.startswith("enc.")}, manifest)
        save(args.out, f"transformer_{tag}.safetensors",
             {k: v for k, v in t.items()
              if k not in ("input",) and not k.startswith("enc.")}, manifest)

        # ---- pass 2..: the generator at every w ----------------------------
        outs = {}
        for w in ws:
            taps = Taps()
            tap_generator(model, taps)
            out, _, _ = model(x, w=w, detach_16=True, adain=False)
            taps.off()
            g = dict(taps.t)
            g["output"] = out[0].contiguous()
            g["w"] = torch.tensor([float(w)])
            outs[w] = g["output"]
            print(f"    w={w:.2f}  out[min,max,mean] = [{out.min():.4f}, "
                  f"{out.max():.4f}, {out.mean():.4f}]", flush=True)
            save(args.out, f"gen_{tag}_w{wtag(w)}.safetensors", g, manifest)

        # ---- the dial's DIRECTION, measured, not asserted from the source ---
        # Larger w must move the output monotonically further from the w=0
        # reconstruction and closer to the degraded input's own structure.
        base = outs[min(ws)]
        drift = {w: float((outs[w] - base).abs().max()) for w in ws}
        print("    max|out(w) - out(w=0)| :",
              ", ".join(f"w={w:.2f}->{d:.4f}" for w, d in sorted(drift.items())), flush=True)
        ordered = [drift[w] for w in sorted(ws)]
        assert all(a <= b + 1e-6 for a, b in zip(ordered, ordered[1:])), \
            f"w does not monotonically increase the CFT contribution: {drift}"
        if 0.0 in drift:
            assert drift[0.0] == 0.0, "w=0 must be the no-fusion baseline"
        manifest[f"w_drift_{tag}"] = {f"{w:.2f}": drift[w] for w in sorted(ws)}

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"  manifest -> {os.path.join(args.out, 'manifest.json')}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
