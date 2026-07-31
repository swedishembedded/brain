#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump CodeFormer / VQGAN reference goldens for brain's `crates/vqgan` parity tests.

Runs the ORIGINAL CodeFormer `basicsr.archs.vqgan_arch.VQAutoEncoder` (CPU,
fp32, fixed seed) and captures every stage boundary with forward hooks during a
real `model(x)` call — stage boundaries are where a port breaks, and one
end-to-end golden cannot bisect a failure.

The VQGAN core is: Encoder (25 blocks) -> VectorQuantizer (nearest, 1024x256)
-> Generator (25 blocks). Two checkpoints carry that exact module trio:
  * vqgan_code1024.pth — the standalone VQGAN (`params_ema`, 329 tensors)
  * codeformer.pth     — CodeFormer (`params_ema`, 515 tensors); its
                         encoder./quantize./generator. prefixes are the same
                         modules with a stage-III finetuned encoder.

Per variant it writes (all tensors f32, indices I32 — brain's safetensors
reader is F32/F16/BF16/I32/I64):

  codebook.safetensors      quantize.embedding.weight (1024, 256)
  quantizer.safetensors     standalone VQ unit goldens: fixed random z at two
                            spatial sizes -> full distance matrix d, argmin
                            indices, min distance, z_q. Gates the vq_argmin
                            kernel dispatch WITHOUT the encoder.
  stages_128.safetensors    128x128 input: EVERY encoder block output (enc.00
                            ..enc.24), the quantizer, EVERY generator block
                            output (gen.00..gen.24), the final image, plus
                            sub-block taps inside one ResBlock and one AttnBlock.
                            Small enough to tap exhaustively; full bisection.
  e2e_512_synth.safetensors 512x512 synthetic input at the REAL config: the
  e2e_512_face.safetensors  512x512 real aligned face (realistic codebook
                            occupancy). Taps the CodeFormer fuse points
                            (encoder 2/5/8/11/14/18 + 24, generator
                            6/9/12/15/18/21 + 24) so the CodeFormer follow-up
                            needs no re-dump.
  manifest.json             per-file sha256 + every tensor's shape and dtype,
                            the block topology (index -> class -> out shape),
                            run params, and library versions.

usage:
  python3 tools/codeformer_dump_reference.py \
      --code   /path/to/CodeFormer \
      --weights /path/to/weights/codeformer \
      --out    testdata/restore/vqgan
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

# VQAutoEncoder(img_size, nf, ch_mult, quantizer, res_blocks, attn_resolutions,
#               codebook_size) exactly as CodeFormer.__init__ constructs it
# (codeformer_arch.py:166). emb_dim defaults to 256.
VQ_CONFIG = dict(img_size=512, nf=64, ch_mult=[1, 2, 2, 4, 4, 8],
                 quantizer="nearest", res_blocks=2, attn_resolutions=[16],
                 codebook_size=1024, emb_dim=256, beta=0.25)

# CodeFormer's controllable-feature-transformation taps (codeformer_arch.py:204).
# Dumped now so the follow-up workflow does not need a second reference run.
FUSE_ENC = [2, 5, 8, 11, 14, 18]   # '512','256','128','64','32','16'
FUSE_GEN = [6, 9, 12, 15, 18, 21]  # '16','32','64','128','256','512'

# Sub-block taps: one ResBlock and one AttnBlock, to bisect INSIDE a block.
SUB_RESBLOCK = 1    # encoder block 1: ResBlock(64 -> 64), no conv_out shortcut
SUB_RESBLOCK_SC = 4  # encoder block 4: ResBlock(64 -> 128), has conv_out
SUB_ATTNBLOCK = 17  # encoder block 17: AttnBlock(512) at the latent resolution


def load_vqgan_arch(code_dir):
    """Import CodeFormer's vqgan_arch.py directly.

    `import basicsr` pulls in losses -> lpips, which is not installed and is
    irrelevant here. Stub the two names vqgan_arch actually imports and exec the
    module file, so the math is byte-for-byte the reference's own.
    """
    utils = types.ModuleType("basicsr.utils")
    utils.get_root_logger = lambda *a, **k: __import__("logging").getLogger("vqgan")
    registry = types.ModuleType("basicsr.utils.registry")

    class _Reg:
        def register(self, cls=None, **k):
            return cls if cls is not None else (lambda c: c)
    registry.ARCH_REGISTRY = _Reg()
    utils.registry = registry
    pkg = types.ModuleType("basicsr")
    pkg.utils = utils
    for name, mod in [("basicsr", pkg), ("basicsr.utils", utils),
                      ("basicsr.utils.registry", registry)]:
        sys.modules[name] = mod

    path = os.path.join(code_dir, "basicsr", "archs", "vqgan_arch.py")
    spec = importlib.util.spec_from_file_location("vqgan_arch", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vqgan_arch"] = mod
    spec.loader.exec_module(mod)
    return mod


def build(arch, ckpt_path):
    """Build VQAutoEncoder and load the encoder/quantize/generator weights."""
    m = arch.VQAutoEncoder(VQ_CONFIG["img_size"], VQ_CONFIG["nf"],
                           VQ_CONFIG["ch_mult"], VQ_CONFIG["quantizer"],
                           VQ_CONFIG["res_blocks"], VQ_CONFIG["attn_resolutions"],
                           VQ_CONFIG["codebook_size"], VQ_CONFIG["emb_dim"],
                           VQ_CONFIG["beta"])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "params_ema" in ck, f"{ckpt_path}: expected 'params_ema', got {list(ck)}"
    sd = ck["params_ema"]
    # codeformer.pth carries the transformer/fuse tensors too; take only the
    # three VQGAN modules and require an EXACT match for those (never zero-fill).
    keep = {k: v for k, v in sd.items()
            if k.split(".")[0] in ("encoder", "quantize", "generator")}
    missing, unexpected = m.load_state_dict(keep, strict=False)
    assert not missing, f"missing VQGAN tensors: {missing[:8]}"
    assert not unexpected, f"unexpected VQGAN tensors: {unexpected[:8]}"
    dropped = sorted({k.split(".")[0] for k in sd if k not in keep})
    return m.float().eval(), len(keep), len(sd), dropped


def det_image(h, w, seed):
    """Deterministic RGB test pattern in [-1, 1], shape (1, 3, h, w).

    A smooth pattern plus seeded noise: pure gradients are too forgiving of
    channel/layout bugs, noise alone gives a degenerate codebook occupancy.
    """
    ys = torch.linspace(0, 3.14159, h).unsqueeze(1).expand(h, w)
    xs = torch.linspace(0, 6.28318, w).unsqueeze(0).expand(h, w)
    r = torch.sin(3.0 * xs + ys)
    g = torch.cos(2.0 * xs) * torch.sin(0.5 * ys)
    b = 2.0 * (ys / 3.14159) - 1.0
    img = torch.stack([r, g, b], 0)
    g = torch.Generator().manual_seed(seed)
    img = img + 0.15 * torch.randn(img.shape, generator=g)
    return img.clamp(-1.0, 1.0).unsqueeze(0).contiguous()


def face_image(path):
    """A real 512x512 aligned face -> (1, 3, 512, 512) in [-1, 1], BGR-free RGB."""
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("RGB")
    a = torch.from_numpy(np.asarray(im).copy()).float() / 255.0   # (H, W, 3)
    return (a.permute(2, 0, 1) * 2.0 - 1.0).unsqueeze(0).contiguous()


class Taps:
    """Forward-hook capture. Names are the golden's tensor names.

    Every run here is batch 1, and the leading batch axis is stripped so a
    golden's shape is the (C, H, W) the Rust side actually compares.
    """

    def __init__(self):
        self.t, self.h = {}, []

    def on(self, module, name, pick=lambda o: o):
        def f(_m, _i, out):
            self.t[name] = pick(out)[0].detach().float().clone().contiguous()
        self.h.append(module.register_forward_hook(f))

    def off(self):
        for h in self.h:
            h.remove()
        self.h = []


def block_topology(blocks):
    return [type(b).__name__ for b in blocks]


def run(model, x, enc_idx, gen_idx, sub_taps):
    """One real VQAutoEncoder forward with hooks on the requested blocks."""
    taps = Taps()
    for i in enc_idx:
        taps.on(model.encoder.blocks[i], f"enc.{i:02d}")
    for i in gen_idx:
        taps.on(model.generator.blocks[i], f"gen.{i:02d}")
    taps.on(model.quantize, "quant.z_q", pick=lambda o: o[0])
    if sub_taps:
        for bi, tag in ((SUB_RESBLOCK, "res"), (SUB_RESBLOCK_SC, "res_sc")):
            b = model.encoder.blocks[bi]
            taps.on(b.norm1, f"sub.{tag}{bi:02d}.norm1")
            taps.on(b.conv1, f"sub.{tag}{bi:02d}.conv1")
            taps.on(b.norm2, f"sub.{tag}{bi:02d}.norm2")
            taps.on(b.conv2, f"sub.{tag}{bi:02d}.conv2")
            if hasattr(b, "conv_out"):
                taps.on(b.conv_out, f"sub.{tag}{bi:02d}.conv_out")
        a = model.encoder.blocks[SUB_ATTNBLOCK]
        for n in ("norm", "q", "k", "v", "proj_out"):
            taps.on(getattr(a, n), f"sub.attn{SUB_ATTNBLOCK:02d}.{n}")

    with torch.no_grad():
        out, cb_loss, stats = model(x)
    taps.off()

    t = dict(taps.t)
    t["input"] = x[0].detach().float().contiguous()
    t["output"] = out[0].detach().float().contiguous()
    t["codebook_loss"] = cb_loss.detach().float().reshape(1)
    t["perplexity"] = stats["perplexity"].detach().float().reshape(1)
    t["mean_distance"] = stats["mean_distance"].detach().float().reshape(1)
    idx = stats["min_encoding_indices"].reshape(-1)
    t["indices"] = idx.to(torch.int32)
    return t, out


def direct_sqdist(z, emb, chunk=32):
    """||z - e||^2 accumulated directly, the way brain's vq_argmin kernel does."""
    outs = []
    for i in range(0, z.shape[0], chunk):
        diff = z[i:i + chunk].unsqueeze(1) - emb.unsqueeze(0)   # (m, K, D)
        outs.append((diff * diff).sum(-1))
    return torch.cat(outs, 0)


def add_distance_golden(t, model, lq_feat, prefix="vq"):
    """Recompute the VQ assignment independently and self-validate.

    The reference expands ||z-e||^2 = |z|^2 + |e|^2 - 2 z.e; brain's `vq_argmin`
    kernel accumulates the squared difference directly. Both forms are dumped so
    the port can be checked against either, and the disagreement (if any) is
    reported rather than hidden — a near-tie that flips is a real porting risk.
    """
    emb = model.quantize.embedding.weight            # (K, D)
    z = lq_feat.permute(0, 2, 3, 1).reshape(-1, emb.shape[1])   # (M, D)
    d_expanded = ((z ** 2).sum(1, keepdim=True) + (emb ** 2).sum(1)
                  - 2.0 * (z @ emb.t()))
    with torch.no_grad():
        d_direct = direct_sqdist(z, emb)
    ai, ad = d_expanded.argmin(1), d_direct.argmin(1)
    n_flip = int((ai != ad).sum().item())
    print(f"    expanded vs direct sq-distance: {n_flip}/{z.shape[0]} argmin "
          f"flips, max|d_exp - d_dir| = {(d_expanded - d_direct).abs().max():.3e}",
          flush=True)
    assert torch.equal(ai.to(torch.int32), t["indices"]), \
        "recomputed argmin != reference min_encoding_indices"
    t[f"{prefix}.z_flat"] = z.contiguous()
    t[f"{prefix}.d"] = d_expanded.contiguous()
    t[f"{prefix}.d_direct"] = d_direct.contiguous()
    t[f"{prefix}.min_dist"] = d_expanded.min(1).values.contiguous()
    t[f"{prefix}.indices_direct"] = ad.to(torch.int32)
    t[f"{prefix}.n_unique_codes"] = torch.tensor([float(ai.unique().numel())])
    t[f"{prefix}.n_argmin_flips"] = torch.tensor([float(n_flip)])
    return t


def quantizer_unit(model, seed):
    """Codebook-lookup goldens with NO encoder in the loop."""
    emb = model.quantize.embedding.weight
    out = {"codebook": emb.detach().float().contiguous()}
    for hw in (4, 16):
        g = torch.Generator().manual_seed(seed + hw)
        # scaled to the codebook's own magnitude so the argmin is not trivial
        z = torch.randn(1, emb.shape[1], hw, hw, generator=g) * emb.std() * 3.0
        with torch.no_grad():
            z_q, loss, stats = model.quantize(z)
        zf = z.permute(0, 2, 3, 1).reshape(-1, emb.shape[1])
        d = ((zf ** 2).sum(1, keepdim=True) + (emb ** 2).sum(1) - 2.0 * (zf @ emb.t()))
        assert torch.equal(d.argmin(1).to(torch.int32),
                           stats["min_encoding_indices"].reshape(-1).to(torch.int32))
        p = f"u{hw}"
        out[f"{p}.z"] = z[0].contiguous()
        out[f"{p}.z_flat"] = zf.contiguous()
        out[f"{p}.d"] = d.contiguous()
        out[f"{p}.indices"] = stats["min_encoding_indices"].reshape(-1).to(torch.int32)
        out[f"{p}.min_dist"] = d.min(1).values.contiguous()
        out[f"{p}.z_q"] = z_q[0].detach().float().contiguous()
        # get_codebook_feat is the decode-side lookup CodeFormer actually uses.
        # It returns the RAW gathered codebook vector; quantize.forward returns
        # the straight-through form `z + (z_q - z)`, which is the same value in
        # exact arithmetic but NOT bit-identical in fp32. Both are dumped.
        with torch.no_grad():
            feat = model.quantize.get_codebook_feat(
                stats["min_encoding_indices"].reshape(-1, 1),
                shape=[1, hw, hw, emb.shape[1]])
        st_gap = (feat - z_q).abs().max().item()
        print(f"    u{hw}: straight-through z_q vs raw gather max|d| = {st_gap:.3e}",
              flush=True)
        assert st_gap < 1e-4, f"get_codebook_feat diverges from quantize z_q ({st_gap})"
        out[f"{p}.codebook_feat"] = feat[0].detach().float().contiguous()
    return out


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
    save_file(tensors, path, metadata={"src": "CodeFormer vqgan_arch fp32 CPU"})
    manifest[name] = {
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                    for k, v in sorted(tensors.items())},
    }
    print(f"  wrote {name}  ({len(tensors)} tensors, "
          f"{os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="CodeFormer repo root")
    ap.add_argument("--weights", required=True, help="dir holding the .pth files")
    ap.add_argument("--out", required=True, help="golden output root")
    ap.add_argument("--face", default=None, help="512x512 aligned face png")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--variants", default="vqgan_code1024,codeformer")
    args = ap.parse_args()

    arch = load_vqgan_arch(args.code)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    root_manifest = {
        "config": VQ_CONFIG,
        "seed": args.seed,
        "reference": {
            "arch": "basicsr.archs.vqgan_arch.VQAutoEncoder (CodeFormer)",
            "code": os.path.abspath(args.code),
            "device": "cpu", "dtype": "float32",
        },
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "fuse_taps": {"encoder": FUSE_ENC, "generator": FUSE_GEN},
        "notes": [
            "swish(x) = x*sigmoid(x) (SiLU); normalize = GroupNorm(32, eps=1e-6, affine)",
            "Downsample = F.pad(x,(0,1,0,1),'constant',0) then Conv2d(k3,s2,p0) — "
            "asymmetric right/bottom padding, NOT symmetric",
            "Upsample = F.interpolate(scale_factor=2, mode='nearest') then Conv2d(k3,s1,p1) "
            "— NOT a ConvTranspose",
            "AttnBlock is single-head over H*W tokens, scale = c**-0.5 with c = channels; "
            "softmax over the KEY axis (dim=2 of q^T k)",
            "VectorQuantizer distance is expanded (|z|^2 + |e|^2 - 2 z.e); ties -> lowest index",
            "quantize.forward returns the straight-through form z + (z_q - z), which is "
            "NOT bit-identical to the raw codebook gather; get_codebook_feat returns the "
            "raw gather. `quant.z_q` is the straight-through form, `<u>.codebook_feat` "
            "the raw one.",
        ],
    }

    for variant in args.variants.split(","):
        variant = variant.strip()
        ckpt = os.path.join(args.weights, f"{variant}.pth")
        if not os.path.exists(ckpt):
            print(f"SKIP {variant}: {ckpt} not found", flush=True)
            continue
        print(f"=== {variant} ({ckpt})", flush=True)
        model, n_used, n_total, dropped = build(arch, ckpt)
        out_dir = os.path.join(args.out, variant)
        os.makedirs(out_dir, exist_ok=True)
        manifest = dict(root_manifest)
        manifest["checkpoint"] = {
            "path": os.path.abspath(ckpt),
            "sha256": sha256_file(ckpt),
            "key": "params_ema",
            "tensors_total": n_total, "tensors_used": n_used,
            "prefixes_dropped": dropped,
        }
        manifest["topology"] = {
            "encoder": block_topology(model.encoder.blocks),
            "generator": block_topology(model.generator.blocks),
        }
        print(f"  loaded {n_used}/{n_total} tensors "
              f"(dropped prefixes: {dropped or 'none'})", flush=True)

        save(out_dir, "codebook.safetensors",
             {"embedding": model.quantize.embedding.weight.detach()}, manifest)
        save(out_dir, "quantizer.safetensors",
             quantizer_unit(model, args.seed), manifest)

        # --- 128x128: EVERY block tapped (full bisection) -------------------
        print("  128x128 full-stage run ...", flush=True)
        x = det_image(128, 128, args.seed)
        t, _ = run(model, x, range(len(model.encoder.blocks)),
                   range(len(model.generator.blocks)), sub_taps=True)
        add_distance_golden(t, model, t["enc.24"].unsqueeze(0))
        save(out_dir, "stages_128.safetensors", t, manifest)

        # --- 512x512: the real config, fuse-point taps ----------------------
        enc_idx = FUSE_ENC + [24]
        gen_idx = FUSE_GEN + [24]
        cases = [("synth", det_image(512, 512, args.seed))]
        if args.face:
            cases.append(("face", face_image(args.face)))
        for tag, xx in cases:
            print(f"  512x512 '{tag}' run ...", flush=True)
            t, _ = run(model, xx, enc_idx, gen_idx, sub_taps=False)
            add_distance_golden(t, model, t["enc.24"].unsqueeze(0))
            print(f"    codes used: {int(t['vq.n_unique_codes'].item())}/1024, "
                  f"perplexity {t['perplexity'].item():.2f}, "
                  f"out[min,max,mean] = [{t['output'].min():.4f}, "
                  f"{t['output'].max():.4f}, {t['output'].mean():.4f}]", flush=True)
            save(out_dir, f"e2e_512_{tag}.safetensors", t, manifest)

        if args.face:
            manifest["face_image"] = os.path.abspath(args.face)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1, sort_keys=True)
        print(f"  manifest -> {os.path.join(out_dir, 'manifest.json')}", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
