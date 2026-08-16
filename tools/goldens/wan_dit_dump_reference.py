#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Wan2.1 diffusion-transformer reference goldens for brain parity tests.

Runs the OFFICIAL `Wan2.1/wan/modules/model.py` (CPU, fp32) and dumps every
boundary brain's DiT has to reproduce, for two configurations:

  dit_tiny.safetensors   a toy-width WanModel with its OWN weights in the file,
                         so the Rust smoke test needs no checkpoint at all
  dit_1_3b.safetensors   the real Wan2.1-T2V-1.3B transformer on a short clip
  manifest.json          shapes, sha256 per file, run params, versions

Both files carry: the input latent, the text encoding, the timestep, the RoPE
tables, `e`/`e0` (the timestep embedding and its 6-way projection), the six
per-block modulation vectors of block 0, block 0's internal taps, EVERY block's
output, the head's two modulation vectors, and the final unpatchified output.

The reference module is imported BY FILE PATH: `import wan.modules.model`
executes `wan/__init__.py`, which drags in the whole model stack (and needs
packages that are not installed here).

## The two independent paths, asserted before a byte is written

1. `wan/modules/model.py`'s `WanModel` - the math authority.
2. diffusers' `WanTransformer3DModel` - a from-scratch reimplementation with
   different module names, a different RoPE formulation (real `cos`/`sin`
   tables with `repeat_interleave` instead of `torch.polar` complex products)
   and its own attention call.

The 1.3B weights ship in the diffusers name space, so the dumper converts them
to the reference names and asserts the two models agree. That conversion is the
same mapping `crates/wan/src/import.rs` implements, so a mistake in it fails
here, in Python, instead of surfacing as a cosine deficit many layers into a
Rust parity run.

## The one shim, and why it does not weaken the golden

`model.py` calls `flash_attention`, which opens with
`assert q.device.type == 'cuda'`; its own non-flash fallback would run
`scaled_dot_product_attention` in **bfloat16** and drop the key-padding mask
entirely. Neither is acceptable for an f32 golden, so `flash_attention` is
replaced with an fp32 SDPA that honours `k_lens`. The replacement is asserted
against the diffusers model, which computes its attention independently.

## The padding experiment (settled here, not by opinion)

`WanModel.forward` zero-pads the token sequence to `seq_len` and passes the
true lengths as `k_lens`. `text2video.py` computes `seq_len` as exactly the
token count (`sp_size = 1`), so the pad is empty on the real path - but the
dumper runs one extra forward at `seq_len = tokens + 37` and asserts the
content rows are unchanged, which is what lets brain compute content rows only
and never carry a mask.

Usage:
  python tools/goldens/wan_dit_dump_reference.py \\
      --transformer /path/to/Wan2.1-T2V-1.3B-Diffusers/transformer \\
      --out testdata/golden/wan/dit [--frames 3 --height 60 --width 104]
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

# `wan/configs/wan_t2v_1_3B.py` - the architecture half of the 1.3B variant.
CFG_1_3B = dict(model_type="t2v", patch_size=(1, 2, 2), text_len=512, in_dim=16,
                dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
                num_heads=12, num_layers=30, qk_norm=True, cross_attn_norm=True,
                eps=1e-6)

# A toy config with the SAME topology: every step kind runs, in a second.
# `dim` is a multiple of 64 because brain's sliced dispatches need each row
# offset to land on the backend's 256-byte storage-binding alignment.
CFG_TINY = dict(model_type="t2v", patch_size=(1, 2, 2), text_len=16, in_dim=4,
                dim=64, ffn_dim=128, freq_dim=32, text_dim=48, out_dim=4,
                num_heads=2, num_layers=3, qk_norm=True, cross_attn_norm=True,
                eps=1e-6)


def load_reference(model_path):
    """Import `wan/modules/model.py` by path, with its siblings resolvable."""
    root = os.path.dirname(os.path.abspath(model_path))
    pkg = types.ModuleType("wan_ref")
    pkg.__path__ = [root]
    sys.modules["wan_ref"] = pkg

    def load(name, path):
        spec = importlib.util.spec_from_file_location(f"wan_ref.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"wan_ref.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    load("attention", os.path.join(root, "attention.py"))
    mod = load("model", model_path)
    mod.flash_attention = cpu_attention
    return mod


def cpu_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                  softmax_scale=None, q_scale=None, causal=False,
                  window_size=(-1, -1), deterministic=False,
                  dtype=torch.bfloat16, version=None):
    """fp32 stand-in for `wan/modules/attention.py:flash_attention`.

    q/k/v are `[B, L, N, C]`. Upstream's own CPU path casts to bf16 and warns
    that it is dropping the padding mask; both would corrupt an f32 golden, so
    the mask is built explicitly and everything stays fp32.
    """
    assert not causal and tuple(window_size) == (-1, -1), "unsupported by this shim"
    assert q_lens is None and q_scale is None and dropout_p == 0.0
    qq, kk, vv = (t.transpose(1, 2).float() for t in (q, k, v))
    mask = None
    if k_lens is not None:
        keep = torch.arange(k.shape[1])[None, :] < k_lens.to(torch.long)[:, None]
        mask = torch.zeros(k.shape[0], 1, 1, k.shape[1])
        mask.masked_fill_(~keep[:, None, None, :], float("-inf"))
    out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


# --------------------------------------------------------------------------
# diffusers <-> reference tensor names (the mapping brain's importer implements)
# --------------------------------------------------------------------------

# Leaf renames that apply inside every block, longest prefix first.
BLOCK_LEAVES = [
    ("attn1.to_q.", "self_attn.q."),
    ("attn1.to_k.", "self_attn.k."),
    ("attn1.to_v.", "self_attn.v."),
    ("attn1.to_out.0.", "self_attn.o."),
    ("attn1.norm_q.", "self_attn.norm_q."),
    ("attn1.norm_k.", "self_attn.norm_k."),
    ("attn2.to_q.", "cross_attn.q."),
    ("attn2.to_k.", "cross_attn.k."),
    ("attn2.to_v.", "cross_attn.v."),
    ("attn2.to_out.0.", "cross_attn.o."),
    ("attn2.norm_q.", "cross_attn.norm_q."),
    ("attn2.norm_k.", "cross_attn.norm_k."),
    ("attn2.add_k_proj.", "cross_attn.k_img."),
    ("attn2.add_v_proj.", "cross_attn.v_img."),
    ("attn2.norm_added_k.", "cross_attn.norm_k_img."),
    ("ffn.net.0.proj.", "ffn.0."),
    ("ffn.net.2.", "ffn.2."),
    # diffusers `norm2` is the cross-attention norm, which upstream calls
    # `norm3`; diffusers `norm3` is the FFN pre-norm, upstream's `norm2`.
    # The two names are SWAPPED, which is exactly the kind of collision that
    # imports cleanly and produces subtly wrong video.
    ("norm2.", "norm3."),
    ("scale_shift_table", "modulation"),
]

TOP_LEVEL = [
    ("patch_embedding.", "patch_embedding."),
    ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
    ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
    ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
    ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
    ("condition_embedder.time_proj.", "time_projection.1."),
    ("condition_embedder.image_embedder.norm1.", "img_emb.proj.0."),
    ("condition_embedder.image_embedder.ff.net.0.proj.", "img_emb.proj.1."),
    ("condition_embedder.image_embedder.ff.net.2.", "img_emb.proj.3."),
    ("condition_embedder.image_embedder.norm2.", "img_emb.proj.4."),
    ("condition_embedder.image_embedder.pos_embed", "img_emb.emb_pos"),
    ("proj_out.", "head.head."),
    ("scale_shift_table", "head.modulation"),
]


def diffusers_to_native(name):
    """One diffusers `WanTransformer3DModel` name -> its reference name."""
    if name.startswith("blocks."):
        i, rest = name[len("blocks."):].split(".", 1)
        for d, n in BLOCK_LEAVES:
            if rest.startswith(d):
                return f"blocks.{i}.{n}{rest[len(d):]}"
        return None
    for d, n in TOP_LEVEL:
        if name.startswith(d):
            return f"{n}{name[len(d):]}"
    return None


def convert(sd):
    out = {}
    for k, v in sd.items():
        n = diffusers_to_native(k)
        assert n is not None, f"unmapped diffusers tensor {k}"
        assert n not in out, f"two source tensors map to {n}"
        out[n] = v
    return out


# --------------------------------------------------------------------------
# deterministic inputs
# --------------------------------------------------------------------------

def det_latent(c, f, h, w, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(c, f, h, w, generator=g)


def det_context(n, d, seed):
    g = torch.Generator().manual_seed(seed + 1)
    return 0.5 * torch.randn(n, d, generator=g)


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous()
               for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h,
                      "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    total = sum(v.numel() for v in tensors.values()) * 4 / 1e6
    print(f"wrote {name}: {len(tensors)} tensors, {total:.1f} MB", flush=True)


def agree(label, a, b, tol=2e-5):
    """Two paths, one assert - relative to the tensor's own scale."""
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    cos = F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()
    print(f"  self-validate {label}: max_abs {d:.3e} / scale {scale:.3g} = {rel:.2e} "
          f"(tol {tol:g}), cosine {cos:.10f}", flush=True)
    assert rel < tol, f"{label}: the two paths disagree by {rel:.3e} relative"


# --------------------------------------------------------------------------
# hooked taps
# --------------------------------------------------------------------------

class Taps:
    """Forward hooks capturing module outputs exactly as the forward made them."""

    def __init__(self):
        self.acc, self.handles = {}, []

    def watch(self, name, module, pick=lambda o: o):
        def hook(_m, _i, o):
            self.acc[name] = pick(o).detach().clone()
        self.handles.append(module.register_forward_hook(hook))

    def watch_input(self, name, module, idx=0):
        def hook(_m, i, _o):
            self.acc[name] = i[idx].detach().clone()
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def block0_taps(model, taps):
    """Every intermediate of block 0 that brain records as a separate step."""
    b = model.blocks[0]
    taps.watch("b0.norm1", b.norm1)
    taps.watch("b0.self_attn.q", b.self_attn.q)
    taps.watch("b0.self_attn.k", b.self_attn.k)
    taps.watch("b0.self_attn.v", b.self_attn.v)
    taps.watch("b0.self_attn.norm_q", b.self_attn.norm_q)
    taps.watch("b0.self_attn.norm_k", b.self_attn.norm_k)
    taps.watch("b0.self_attn.o", b.self_attn.o)
    taps.watch_input("b0.self_attn.o_in", b.self_attn.o)
    taps.watch("b0.norm3", b.norm3)
    taps.watch("b0.cross_attn.q", b.cross_attn.q)
    taps.watch("b0.cross_attn.k", b.cross_attn.k)
    taps.watch("b0.cross_attn.v", b.cross_attn.v)
    taps.watch("b0.cross_attn.o", b.cross_attn.o)
    taps.watch_input("b0.cross_attn.o_in", b.cross_attn.o)
    taps.watch("b0.norm2", b.norm2)
    taps.watch("b0.ffn.0", b.ffn[0])
    taps.watch("b0.ffn.1", b.ffn[1])
    taps.watch("b0.ffn.2", b.ffn[2])


def run(model, latent, context, t, seq_len):
    """One real forward, with `x`/`context` in the shapes the pipeline uses."""
    ts = torch.tensor([float(t)])
    return model([latent], t=ts, context=[context], seq_len=seq_len)[0]


def rope_tables(model, cfg, f, h, w):
    """The (cos, sin) rows brain's `rope_interleave_table` consumes.

    `WanModel.freqs` is `[1024, head_dim/2]` complex, split per axis exactly as
    `rope_apply` splits it; the per-token row is the concatenation of the three
    axes' entries, which is what `dit::rope::tables_for_ids` builds from
    `(f, h, w)` ids.
    """
    c = cfg["dim"] // cfg["num_heads"] // 2
    parts = model.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    rows = torch.cat([
        parts[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, -1)
    return rows.real.float().contiguous(), rows.imag.float().contiguous()


def dump(model, cfg, latent, context, timestep, out_dir, name, manifest,
         diff_model=None, pad_probe=37):
    dim = cfg["dim"]
    pt, ph, pw = cfg["patch_size"]
    c, fr, hh, ww = latent.shape
    grid = (fr // pt, hh // ph, ww // pw)
    tokens = grid[0] * grid[1] * grid[2]
    print(f"\n=== {name}: latent {tuple(latent.shape)} -> grid {grid} = {tokens} tokens, "
          f"context {tuple(context.shape)}, t={timestep} ===", flush=True)

    taps = Taps()
    block0_taps(model, taps)
    for i, b in enumerate(model.blocks):
        taps.watch(f"block.{i}", b)
    taps.watch("patch_embed", model.patch_embedding)
    taps.watch("text_embed", model.text_embedding)
    taps.watch("time_embed", model.time_embedding)
    taps.watch("time_proj", model.time_projection)
    taps.watch("head", model.head)
    taps.watch("head_norm", model.head.norm)
    out = run(model, latent, context, timestep, tokens)
    taps.close()
    acc = dict(taps.acc)

    e = acc["time_embed"].float()                       # [1, dim]
    e0 = acc["time_proj"].float().unflatten(1, (6, dim))  # [1, 6, dim]
    head_mod = (model.head.modulation + e.unsqueeze(1)).float()[0]  # [2, dim]

    cos, sin = rope_tables(model, cfg, *grid)

    tensors = {
        "latent": latent,
        "context": context,
        "timestep": torch.tensor([float(timestep)]),
        "grid": torch.tensor([float(g) for g in grid]),
        "rope_cos": cos,
        "rope_sin": sin,
        "e": e[0],
        "e0": e0[0],
        "head_mod": head_mod,
        "patch_embed": acc["patch_embed"][0].flatten(1).transpose(0, 1).contiguous(),
        "text_embed": acc["text_embed"][0],
        "head_norm": acc["head_norm"][0],
        "head_out": acc["head"][0],
        "out": out,
    }
    for i in range(cfg["num_layers"]):
        tensors[f"block.{i}"] = acc[f"block.{i}"][0]
    for k, v in acc.items():
        if k.startswith("b0."):
            tensors[f"tap_{k}"] = v[0] if v.dim() == 3 else v
    for i, b in enumerate(model.blocks):
        tensors[f"mod.{i}"] = (b.modulation + e0).float()[0]

    # ---- self-validation 1: the padding experiment -----------------------
    # The tolerance is fp32 reassociation, not semantics: SDPA blocks the key
    # axis differently at a different sequence length, and 30 residual blocks
    # accumulate that. A real mask defect is not a 1e-5 effect - dropping the
    # mask entirely moves this by orders of magnitude more.
    padded = run(model, latent, context, timestep, tokens + pad_probe)
    agree(f"{name} seq_len pad (+{pad_probe})", padded, out, tol=1e-4)

    # ---- self-validation 2: the independent diffusers model --------------
    if diff_model is not None:
        d_out = diff_model(
            hidden_states=latent.unsqueeze(0),
            timestep=torch.tensor([float(timestep)]),
            encoder_hidden_states=pad_context(context, cfg["text_len"]).unsqueeze(0),
            return_dict=False,
        )[0][0]
        agree(f"{name} diffusers vs reference", d_out, out, tol=5e-5)
        tensors["out_diffusers"] = d_out

    save(out_dir, name, tensors, manifest)
    return tokens


def pad_context(context, text_len):
    """`WanModel.forward` zero-pads each context to `text_len` before embedding."""
    if context.shape[0] >= text_len:
        return context[:text_len]
    return torch.cat([context, context.new_zeros(text_len - context.shape[0], context.shape[1])])


def build_tiny(ref, seed):
    torch.manual_seed(seed)
    m = ref.WanModel(**CFG_TINY)
    # `init_weights` zeroes `head.head.weight`, which would make the whole
    # golden output identically zero and prove nothing. Re-randomise it.
    torch.nn.init.normal_(m.head.head.weight, std=0.02)
    m.eval().requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="scratchpad/reference/wan/Wan2.1/wan/modules/model.py")
    ap.add_argument("--transformer", default=None,
                    help="Wan2.1-T2V-1.3B-Diffusers/transformer directory (real weights)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=3, help="latent frames for the 1.3B run")
    ap.add_argument("--height", type=int, default=60, help="latent height for the 1.3B run")
    ap.add_argument("--width", type=int, default=104, help="latent width for the 1.3B run")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timestep", type=float, default=750.0)
    ap.add_argument("--skip-tiny", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    ref = load_reference(args.reference)
    print(f"reference: {args.reference}", flush=True)

    manifest = {"run": {"seed": args.seed, "timestep": args.timestep,
                        "tiny_config": {k: list(v) if isinstance(v, tuple) else v
                                        for k, v in CFG_TINY.items()},
                        "config_1_3b": {k: list(v) if isinstance(v, tuple) else v
                                        for k, v in CFG_1_3B.items()},
                        "shims": ["flash_attention -> fp32 SDPA honouring k_lens"]},
                "versions": {"torch": torch.__version__, "python": sys.version.split()[0]}}

    if not args.skip_tiny:
        m = build_tiny(ref, args.seed)
        lat = det_latent(CFG_TINY["in_dim"], 5, 16, 16, args.seed)
        ctx = det_context(11, CFG_TINY["text_dim"], args.seed)
        tokens = dump(m, CFG_TINY, lat, ctx, args.timestep, args.out,
                      "dit_tiny.safetensors", manifest)
        manifest["run"]["tiny_tokens"] = tokens
        # The toy model's own weights, in the REFERENCE name space, so the Rust
        # smoke test is a pure replay with no checkpoint anywhere.
        sd = {k: v for k, v in m.state_dict().items()}
        save(args.out, "dit_tiny_weights.safetensors", sd, manifest)
        del m

    if args.transformer:
        import diffusers
        print(f"loading {args.transformer}", flush=True)
        dm = diffusers.WanTransformer3DModel.from_pretrained(
            args.transformer, torch_dtype=torch.float32)
        dm.eval().requires_grad_(False)
        sd = convert(dm.state_dict())
        m = ref.WanModel(**CFG_1_3B)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing, f"missing after conversion: {missing[:8]}"
        assert not unexpected, f"unexpected after conversion: {unexpected[:8]}"
        print(f"converted {len(sd)} diffusers tensors -> reference names (strict)", flush=True)
        m.eval().requires_grad_(False)

        lat = det_latent(CFG_1_3B["in_dim"], args.frames, args.height, args.width, args.seed)
        ctx = det_context(97, CFG_1_3B["text_dim"], args.seed)
        tokens = dump(m, CFG_1_3B, lat, ctx, args.timestep, args.out,
                      "dit_1_3b.safetensors", manifest, diff_model=dm)
        manifest["run"]["tokens_1_3b"] = tokens
        manifest["run"]["latent_1_3b"] = [args.frames, args.height, args.width]
    else:
        print("no --transformer: skipping the real-weights golden", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
