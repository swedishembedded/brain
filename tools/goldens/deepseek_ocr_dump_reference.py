#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump DeepSeek-OCR reference goldens for brain's `crates/deepseekocr` parity ladder.

The model is DeepSeek-OCR: a **DeepEncoder** (SAM ViT-B tower with windowed and
global attention and decomposed relative position bias -> a conv neck + 16x token
compressor -> CLIP-L whose own patch embed is BYPASSED, the compressor output
being injected as its patch tokens -> a linear projector over
`concat([clip_spatial, compressor_flat])`) feeding a **DeepSeek-V2-family MoE
decoder** (plain MHA, NOT MLA; NEOX-layout RoPE; one leading dense SwiGLU layer
then MoE layers with softmax top-k routing and unweighted shared experts).

Files written under `--out`:

  tiny/ckpt/model.safetensors  a RANDOM whole-pipeline DeepSeek-OCR in miniature
                             (image -> SAM -> compressor -> CLIP -> projector ->
                             MoE decoder -> logits), seeded, needing NO
                             checkpoint. Its dims are chosen so that the
                             coincidences the real model hides cannot mask a
                             port bug: at real scale CLIP width == compressor
                             output channels == 1024 (a concat-order swap is
                             then arithmetically invisible), SAM head_dim == 64
                             == the compressor's channel counts, and the SAM
                             grid, window and head widths collapse onto the same
                             handful of numbers. Here every one of those is a
                             DIFFERENT number, both spatial extents differ
                             (H != W everywhere), and one SAM block's window
                             divides neither grid extent, so the zero-pad path
                             is exercised.
  tiny/golden.safetensors    per-STAGE taps of the forward pass of that model:
                             patch embed, every SAM block plus its internals and
                             its rel-pos intermediates (`Rh`/`Rw` and the bias
                             term computed two independent ways), every
                             compressor stage, every CLIP block plus internals,
                             the vision concat, the projector output, the spliced
                             decoder input, every decoder layer plus internals,
                             the router logits with BOTH gate variants, and the
                             final logits.
  manifest-tiny.json         shapes + sha256 per file, the tiny config, the run
                             parameters and the recorded semantic findings.
                             A run WITH a real checkpoint writes `manifest.json`
                             instead, so a tiny-only run can never clobber a full
                             manifest.

  (not yet implemented, behind `--gguf`/`--mmproj`:)
  real/vision.safetensors    per-stage taps of the real DeepEncoder (SAM blocks,
                             compressor, CLIP blocks, concat, projector) at each
                             supported resolution mode.
  real/decoder.safetensors   per-layer taps of the real MoE decoder plus router
                             logits/gates and the greedy-decoded token ids.
  real/preprocess.safetensors  the preprocessed image tensor and the image-row
                             layout (grid tokens, newlines, separator).

Checkpoint tensor names (`tiny/ckpt/model.safetensors`). There is no consuming
Rust crate yet, so this scheme IS the contract a future `deepseekocr::import`
matches against. It is flat, dot-separated, and follows the natural attribute
path of the reference PyTorch modules (NOT the GGUF's `v.sam.blk.N.*` short
names, which the GGUF loader classifies separately):

  sam.patch_embed.proj.{weight,bias}     Conv2d(in_chans, C, k=p, s=p)
  sam.pos_embed                          [1, gh, gw, C] learned absolute
  sam.blocks.{i}.norm1.{weight,bias}     LayerNorm(C)
  sam.blocks.{i}.attn.qkv.{weight,bias}  [3C, C] fused, q|k|v major
  sam.blocks.{i}.attn.proj.{weight,bias}
  sam.blocks.{i}.attn.rel_pos_h          [Lh, head_dim] learned table
  sam.blocks.{i}.attn.rel_pos_w          [Lw, head_dim]
  sam.blocks.{i}.norm2.{weight,bias}
  sam.blocks.{i}.mlp.lin{1,2}.{weight,bias}
  compressor.neck.0.weight               Conv2d(C, c_mid, k=1), NO bias
  compressor.neck.1.{weight,bias}        LayerNorm2d(c_mid) (channels-first)
  compressor.neck.2.weight               Conv2d(c_mid, c_mid, k=3, p=1), NO bias
  compressor.neck.3.{weight,bias}        LayerNorm2d(c_mid)
  compressor.net_2.weight                Conv2d(c_mid, c2, k=3, s=2, p=1), NO bias
  compressor.net_3.weight                Conv2d(c2, c_out, k=3, s=2, p=1), NO bias
  clip.patch_bypass.{weight,bias}        Linear(c_out, W): the patch-embed
                                         bypass adapter. It exists ONLY because
                                         this fixture forces c_out != clip_width
                                         (the concat-order gate); at real scale
                                         both are 1024, the compressor output is
                                         the patch token verbatim, and this
                                         tensor is ABSENT.
  clip.class_embedding                   [W] prepended cls token
  clip.position_embedding                [1, 1 + gn*gn, W] learned absolute, at
                                         the checkpoint's NATIVE square patch
                                         grid `gn = image_size / patch_size`.
                                         Row 0 is the class token; rows 1.. are
                                         the patch grid and are bicubically
                                         resampled onto the compressor's
                                         (non-square, non-native) token grid.
  clip.pre_norm.{weight,bias}            LayerNorm applied to `[cls; tokens]+pos`
                                         BEFORE block 0. The real mmproj carries
                                         `v.pre_ln` and no post-LN, so the tower
                                         input is normalised and its output is
                                         the last block's hidden state verbatim.
  clip.blocks.{i}.norm{1,2}.{weight,bias}
  clip.blocks.{i}.attn.{q,k,v,out}_proj.{weight,bias}
  clip.blocks.{i}.mlp.fc{1,2}.{weight,bias}
  projector.{weight,bias}                Linear(clip_width + c_out, d_model)
  decoder.embed_tokens.weight            [vocab, d_model]
  decoder.layers.{i}.input_layernorm.weight          RMSNorm gain, no bias
  decoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight NO bias, MHA (kv == q heads)
  decoder.layers.{i}.post_attention_layernorm.weight
  decoder.layers.{i}.mlp.{gate,up,down}_proj.weight  dense SwiGLU (leading layer)
  decoder.layers.{i}.mlp.gate.weight                 router [n_routed, d_model]
  decoder.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight
  decoder.layers.{i}.mlp.shared_experts.{gate,up,down}_proj.weight
  decoder.norm.weight
  decoder.lm_head.weight                 untied from embed_tokens

Everything is CPU + fp32 with fixed seeds; every tensor is stored as f32 (brain's
safetensors reader is F32/F16/BF16-only -- the token-id and expert-index tables
are exactly representable).

Five semantic questions are SETTLED HERE by measurement rather than by argument
(porting-playbook 6), and the answers land in the manifest:

  * **decomposed rel-pos bias** -- the einsum formulation
    (`bhwc,hkc->bhwk` plus `bhwc,wkc->bhwk`, broadcast into a
    [q_h, q_w, k_h, k_w] additive term) is recomputed by an independent,
    fully explicit four-deep loop over (q_h, q_w, k_h, k_w) that dots the query
    against `Rh`/`Rw` directly, and the two are asserted equal. Separately, the
    `F.interpolate(mode="linear")` table resize is recomputed from the raw
    half-pixel rule (`src = (dst + 0.5) * L / M - 0.5`, clamped to >= 0, linear
    between neighbours with edge clamping) and asserted equal, so the Rust port
    has a written-down arithmetic contract instead of a library call to imitate.
    The fixture exercises all three table cases: identity (table length exactly
    `2 * extent - 1`), downsample and upsample.
  * **router renormalization** -- the MoE gate is computed BOTH ways from the
    same logits (raw softmax probabilities of the selected experts vs the same
    values renormalized to sum to 1) and the two are asserted to DIFFER, so the
    fixture cannot be a degenerate case where `norm_topk_prob` is unobservable.
    The layer output is built from the RAW, UN-renormalized gate, which is what
    the real checkpoint's consumer does.
  * **vision concat order** -- after building
    `concat([clip_spatial, compressor_flat], dim=-1)` the two halves are sliced
    back out and asserted equal to their sources, with `clip_width != c_out` so
    the halves have different widths and a swapped concatenation cannot produce
    a plausible tensor.
  * **CLIP position resample** -- the learned table lives on the checkpoint's
    NATIVE square patch grid, and the grid the tower actually runs at is the
    compressor's output grid, which is neither square nor native. The class-token
    row is lifted out, the patch rows are resampled with
    `F.interpolate(mode="bicubic", align_corners=False)` (PyTorch's `a = -0.75`
    cubic-convolution kernel with clamp-to-edge taps), and the class row is put
    back at index 0. This fixture's resample UPsamples the height and DOWNsamples
    the width in one call, so a transposed `(h, w)` cannot survive it, and the
    saved `clip_pos_full` tap pins the result.
  * **window zero-pad** -- block 0's window divides neither grid extent, so the
    partition zero-pads. Because the pad happens AFTER norm1, a pad position's
    input is exactly 0 and its qkv is exactly the qkv BIAS: that is asserted
    against the saved bias, and the partition/unpartition round trip is asserted
    to be the identity. "Pad tokens are inert" is the natural wrong guess and it
    changes every window's softmax denominator.

Usage:
  python3 tools/goldens/deepseek_ocr_dump_reference.py --out testdata/deepseek-ocr
"""

import argparse
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

SEED = 0

# LayerNorm epsilons: the SAM tower and its channels-first LayerNorm2d use 1e-6
# (the SAM repo's `partial(nn.LayerNorm, eps=1e-6)`), CLIP uses 1e-5, the
# decoder's RMSNorm uses 1e-6.
EPS_SAM = 1e-6
EPS_CLIP = 1e-5
EPS_RMS = 1e-6

# The image placeholder token whose embeddings the projector output replaces.
IMAGE_TOKEN_ID = 0

# Tiny dims. Every number that could be confused with another number in the same
# kernel is a DIFFERENT number here; see the module docstring for why.
#
#   SAM      grid 13 x 7, window 4 x 3  -> neither extent divides (pads to 16 x 9)
#            embed 16 = 2 heads x head_dim 8; 8 is not any grid/window extent,
#            and the per-window token count 12 is not the embed width 16.
#   rel-pos  block 0: h table 7 == 2*4-1 (IDENTITY), w table 11 > 2*3-1 (DOWN)
#            block 1: h table 15 < 2*13-1 (UP),      w table 13 == 2*7-1 (IDENT)
#   compress 13 x 7 -> (stride 2) 7 x 4 -> (stride 2) 4 x 2 = 8 CLIP patch tokens
#            channels 16 -> 6 -> 6 -> 9 -> 11
#   CLIP     8 + 1 cls = 9 tokens, width 14 = 2 heads x head_dim 7; the learned
#            position table lives on a NATIVE 3 x 3 grid (10 rows) and is
#            resampled onto 4 x 2 (height UP, width DOWN, in one call)
#   concat   14 (CLIP) + 11 (compressor) = 25 -> projector -> d_model 12
#   decoder  13 tokens (8 of them image), d_model 12 = 3 heads x head_dim 4
#
# Two of these numbers are set by the CONSUMER's kernel ABI rather than by the
# reference, and both are recorded here so that regenerating the fixture cannot
# silently produce something that is not dispatchable:
#
#   * `sam_embed` must be a multiple of 16. brain binds one attention span's
#     `ctx` sliced at `row0 * sam_embed` floats and a storage-binding offset must
#     be 64-float (256 B) aligned; a 4 x 3 window is 12 rows, so 12 * embed must
#     be a multiple of 64, i.e. 16 | embed. (The previous fixture used 10, which
#     is NOT dispatchable -- see `sam1::SamViTConfig::check_bindable`.)
#   * `batch` is 1. The same span-offset rule has to hold across the batch
#     stride as well, and `grid_h * grid_w * sam_embed` generally is not a
#     multiple of 64, so the SAM tower is single-image. Nothing else in the
#     pipeline needs a batch axis to stay honest: every layout question this
#     fixture exists to settle (grid order, concat order, row/column extents) is
#     visible at B = 1 because H != W everywhere.
TINY = {
    "batch": 1,
    "in_chans": 3,
    "image_h": 26,
    "image_w": 14,
    "patch": 2,
    "sam_grid_h": 13,
    "sam_grid_w": 7,
    "sam_embed": 16,
    "sam_heads": 2,
    "sam_head_dim": 8,
    "sam_mlp": 17,
    "sam_blocks": 2,
    # per block: (window_h, window_w); 0 means the block is global (unwindowed)
    "sam_windows": [(4, 3), (0, 0)],
    # per block: (len(rel_pos_h), len(rel_pos_w))
    "sam_rel_len": [(7, 11), (15, 13)],
    "c_mid": 6,
    "c2": 9,
    "c_out": 11,
    "clip_width": 14,
    "clip_heads": 2,
    "clip_head_dim": 7,
    "clip_mlp": 20,
    "clip_blocks": 2,
    # The CLIP checkpoint's own (square) geometry: the learned position table has
    # `1 + (image_size / patch)^2` rows and is resampled onto the compressor grid.
    "clip_patch": 5,
    "clip_image_size": 15,
    "d_model": 12,
    "dec_heads": 3,
    "dec_head_dim": 4,
    "dec_layers": 2,
    "dense_ff": 21,
    "moe_ff": 7,
    "n_routed": 5,
    "top_k": 2,
    "n_shared": 2,
    "vocab": 19,
    "seq": 13,
    "image_pos": 3,
    "rope_base": 10000.0,
}


def save(out_dir, rel, tensors, manifest, extra=None):
    """Save `tensors` as f32 safetensors under `out_dir/rel` and manifest it."""
    tensors = {
        k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()
    }
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    entry = {
        "sha256": sha,
        "bytes": os.path.getsize(path),
        "dtype": "F32",
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
    }
    if extra:
        entry.update(extra)
    manifest["files"][rel] = entry
    print(f"wrote {rel}: {len(tensors)} tensors, {os.path.getsize(path) / 1e6:.3f} MB",
          flush=True)


class Params:
    """The random tiny checkpoint: a flat name -> tensor dict, saved verbatim."""

    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(seed)
        self.t = {}

    def new(self, name, shape, scale=0.4, mean=0.0):
        assert name not in self.t, f"duplicate parameter {name}"
        v = torch.randn(shape, generator=self.g) * scale + mean
        self.t[name] = v
        return v

    def gain(self, name, n):
        """A norm gain, randomised around 1.

        The default init leaves every norm weight at exactly 1.0 and every norm
        bias at 0.0, which hides a dropped or mis-indexed gain; random values do
        not.
        """
        return self.new(name, (n,), scale=0.25, mean=1.0)

    def norm(self, prefix, n):
        self.gain(prefix + ".weight", n)
        self.new(prefix + ".bias", (n,), scale=0.2)

    def __getitem__(self, name):
        return self.t[name]


def linear(x, w, b=None):
    y = x @ w.t()
    return y if b is None else y + b


def layer_norm(x, w, b, eps):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def layer_norm_2d(x, w, b, eps=EPS_SAM):
    """Channels-first LayerNorm over dim 1 of an NCHW tensor (SAM's LayerNorm2d)."""
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + eps)
    return w[:, None, None] * x + b[:, None, None]


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def rms_norm(x, w, eps=EPS_RMS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def softmax_attention(q, k, v, scale, mask=None):
    """[B, T, D] x [B, S, D] -> (probs, ctx); `mask` is additive."""
    scores = (q * scale) @ k.transpose(-2, -1)
    if mask is not None:
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    return probs, probs @ v


# --------------------------------------------------------------------------
# SAM: windowing and decomposed relative position bias
# --------------------------------------------------------------------------


def window_partition(x, wh, ww):
    """[B, H, W, C] -> ([B * nwin, wh, ww, C], (Hp, Wp)), zero-padding bottom/right.

    The pad is applied AFTER norm1, so a padded position's input is exactly zero
    and its qkv is therefore the qkv BIAS, not zero. Anything that treats pad
    tokens as inert is wrong.
    """
    B, H, W, C = x.shape
    ph = (wh - H % wh) % wh
    pw = (ww - W % ww) % ww
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    hp, wp = H + ph, W + pw
    x = x.reshape(B, hp // wh, wh, wp // ww, ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, wh, ww, C)
    return windows, (hp, wp)


def window_unpartition(windows, wh, ww, pad_hw, hw):
    """Inverse of `window_partition`, cropping the pad back off."""
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // ((hp // wh) * (wp // ww))
    x = windows.reshape(b, hp // wh, wp // ww, wh, ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)
    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


def linear_resample(table, out_len):
    """`F.interpolate(mode="linear", align_corners=False)` written out longhand.

    This is the arithmetic contract the Rust port implements: half-pixel source
    mapping `src = (dst + 0.5) * L / M - 0.5` clamped to >= 0, linear blend of
    the two neighbouring rows with the right neighbour clamped to the last row.
    It is asserted equal to torch's own result below, so if either ever moves,
    the dumper fails instead of blessing a table nobody re-derived.
    """
    src_len, chan = table.shape
    scale = src_len / out_len
    out = torch.empty((out_len, chan), dtype=table.dtype)
    for dst in range(out_len):
        src = max(0.0, (dst + 0.5) * scale - 0.5)
        lo = int(math.floor(src))
        w1 = src - lo
        hi = min(lo + 1, src_len - 1)
        out[dst] = table[lo] * (1.0 - w1) + table[hi] * w1
    return out


def get_rel_pos(q_size, k_size, rel_pos):
    """The learned [L, head_dim] table, resized and indexed to [q, k, head_dim]."""
    max_rel_dist = 2 * max(q_size, k_size) - 1
    src_len = rel_pos.shape[0]
    if src_len != max_rel_dist:
        resized = F.interpolate(
            rel_pos.reshape(1, src_len, -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        resized = resized.reshape(-1, max_rel_dist).permute(1, 0)
        case = "downsample" if src_len > max_rel_dist else "upsample"
        # Independent re-derivation of the same resize (see `linear_resample`).
        hand = linear_resample(rel_pos, max_rel_dist)
        # fp32 rounding only: torch fuses the two-tap blend differently, which
        # costs a few ulps on O(1) values. Anything larger is a real disagreement
        # about the source-index rule.
        gap = (hand - resized).abs().max().item()
        assert gap < 1e-5, f"half-pixel resample disagrees with torch: {gap:.3e}"
    else:
        resized = rel_pos
        case = "identity"
        gap = 0.0
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    info = {"table_len": src_len, "max_rel_dist": max_rel_dist, "case": case,
            "resample_max_abs_vs_longhand": gap}
    return resized[relative_coords.long()], info


def rel_pos_bias_einsum(q, rh, rw, q_size, k_size):
    """The reference formulation: [B, q_h * q_w, k_h * k_w] additive bias."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    b, _, dim = q.shape
    r_q = q.reshape(b, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)
    bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return bias.reshape(b, q_h * q_w, k_h * k_w)


def rel_pos_bias_loops(q, rh, rw, q_size, k_size):
    """The same quantity, recomputed by an explicit loop over every (q, k) pair.

    Deliberately shares nothing with `rel_pos_bias_einsum` except `Rh`/`Rw`: it
    indexes and dots by hand, so an einsum subscript transposed between the h
    and w terms (the exact bug this fixture exists to catch) shows up as a
    mismatch rather than as a plausible tensor.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    b, _, dim = q.shape
    r_q = q.reshape(b, q_h, q_w, dim)
    out = torch.zeros((b, q_h, q_w, k_h, k_w), dtype=q.dtype)
    for qh in range(q_h):
        for qw in range(q_w):
            vec = r_q[:, qh, qw, :]
            for kh in range(k_h):
                dot_h = (vec * rh[qh, kh, :]).sum(-1)
                for kw in range(k_w):
                    out[:, qh, qw, kh, kw] = dot_h + (vec * rw[qw, kw, :]).sum(-1)
    return out.reshape(b, q_h * q_w, k_h * k_w)


# --------------------------------------------------------------------------
# The tiny model: parameters, forward, taps
# --------------------------------------------------------------------------


def build_params(cfg):
    p = Params(SEED)
    c = cfg["sam_embed"]
    p.new("sam.patch_embed.proj.weight",
          (c, cfg["in_chans"], cfg["patch"], cfg["patch"]), scale=0.5)
    p.new("sam.patch_embed.proj.bias", (c,), scale=0.2)
    p.new("sam.pos_embed", (1, cfg["sam_grid_h"], cfg["sam_grid_w"], c), scale=0.3)
    for i in range(cfg["sam_blocks"]):
        pre = f"sam.blocks.{i}."
        p.norm(pre + "norm1", c)
        p.new(pre + "attn.qkv.weight", (3 * c, c), scale=0.35)
        p.new(pre + "attn.qkv.bias", (3 * c,), scale=0.3)
        p.new(pre + "attn.proj.weight", (c, c), scale=0.35)
        p.new(pre + "attn.proj.bias", (c,), scale=0.2)
        lh, lw = cfg["sam_rel_len"][i]
        p.new(pre + "attn.rel_pos_h", (lh, cfg["sam_head_dim"]), scale=0.5)
        p.new(pre + "attn.rel_pos_w", (lw, cfg["sam_head_dim"]), scale=0.5)
        p.norm(pre + "norm2", c)
        p.new(pre + "mlp.lin1.weight", (cfg["sam_mlp"], c), scale=0.35)
        p.new(pre + "mlp.lin1.bias", (cfg["sam_mlp"],), scale=0.2)
        p.new(pre + "mlp.lin2.weight", (c, cfg["sam_mlp"]), scale=0.35)
        p.new(pre + "mlp.lin2.bias", (c,), scale=0.2)

    mid, c2, cout = cfg["c_mid"], cfg["c2"], cfg["c_out"]
    p.new("compressor.neck.0.weight", (mid, c, 1, 1), scale=0.5)
    p.norm("compressor.neck.1", mid)
    p.new("compressor.neck.2.weight", (mid, mid, 3, 3), scale=0.3)
    p.norm("compressor.neck.3", mid)
    p.new("compressor.net_2.weight", (c2, mid, 3, 3), scale=0.3)
    p.new("compressor.net_3.weight", (cout, c2, 3, 3), scale=0.3)

    w = cfg["clip_width"]
    # The learned table is on the CHECKPOINT's native square grid, not on the
    # grid this run uses; row 0 is the class token.
    native = cfg["clip_native_grid"]
    # The patch-embed bypass adapter. At real scale c_out == clip_width == 1024
    # and the compressor output IS the patch token, so no such tensor exists;
    # this fixture deliberately breaks that equality (it is what makes a
    # concat-order swap detectable), which leaves the bypass a real widening.
    p.new("clip.patch_bypass.weight", (w, cout), scale=0.35)
    p.new("clip.patch_bypass.bias", (w,), scale=0.2)
    p.new("clip.class_embedding", (w,), scale=0.5)
    p.new("clip.position_embedding", (1, native * native + 1, w), scale=0.3)
    p.norm("clip.pre_norm", w)
    for i in range(cfg["clip_blocks"]):
        pre = f"clip.blocks.{i}."
        p.norm(pre + "norm1", w)
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            p.new(pre + f"attn.{proj}.weight", (w, w), scale=0.3)
            p.new(pre + f"attn.{proj}.bias", (w,), scale=0.2)
        p.norm(pre + "norm2", w)
        p.new(pre + "mlp.fc1.weight", (cfg["clip_mlp"], w), scale=0.3)
        p.new(pre + "mlp.fc1.bias", (cfg["clip_mlp"],), scale=0.2)
        p.new(pre + "mlp.fc2.weight", (w, cfg["clip_mlp"]), scale=0.3)
        p.new(pre + "mlp.fc2.bias", (w,), scale=0.2)

    d = cfg["d_model"]
    p.new("projector.weight", (d, w + cout), scale=0.3)
    p.new("projector.bias", (d,), scale=0.2)

    p.new("decoder.embed_tokens.weight", (cfg["vocab"], d), scale=0.5)
    for i in range(cfg["dec_layers"]):
        pre = f"decoder.layers.{i}."
        p.gain(pre + "input_layernorm.weight", d)
        inner = cfg["dec_heads"] * cfg["dec_head_dim"]
        for proj in ["q_proj", "k_proj", "v_proj"]:
            p.new(pre + f"self_attn.{proj}.weight", (inner, d), scale=0.3)
        p.new(pre + "self_attn.o_proj.weight", (d, inner), scale=0.3)
        p.gain(pre + "post_attention_layernorm.weight", d)
        if i == 0:
            for proj, shape in [("gate_proj", (cfg["dense_ff"], d)),
                                ("up_proj", (cfg["dense_ff"], d)),
                                ("down_proj", (d, cfg["dense_ff"]))]:
                p.new(pre + f"mlp.{proj}.weight", shape, scale=0.3)
        else:
            p.new(pre + "mlp.gate.weight", (cfg["n_routed"], d), scale=0.7)
            for e in range(cfg["n_routed"]):
                for proj, shape in [("gate_proj", (cfg["moe_ff"], d)),
                                    ("up_proj", (cfg["moe_ff"], d)),
                                    ("down_proj", (d, cfg["moe_ff"]))]:
                    p.new(pre + f"mlp.experts.{e}.{proj}.weight", shape, scale=0.4)
            shared = cfg["n_shared"] * cfg["moe_ff"]
            for proj, shape in [("gate_proj", (shared, d)),
                                ("up_proj", (shared, d)),
                                ("down_proj", (d, shared))]:
                p.new(pre + f"mlp.shared_experts.{proj}.weight", shape, scale=0.3)
    p.gain("decoder.norm.weight", d)
    p.new("decoder.lm_head.weight", (cfg["vocab"], d), scale=0.4)
    return p


def sam_forward(image, p, cfg, taps, findings):
    """SAM ViT tower: [B, C_in, H, W] image -> [B, gh, gw, C] tokens."""
    c = cfg["sam_embed"]
    heads, hd = cfg["sam_heads"], cfg["sam_head_dim"]
    x = F.conv2d(image, p["sam.patch_embed.proj.weight"],
                 p["sam.patch_embed.proj.bias"], stride=cfg["patch"])
    x = x.permute(0, 2, 3, 1).contiguous()
    assert x.shape[1:3] == (cfg["sam_grid_h"], cfg["sam_grid_w"]), x.shape
    taps["sam_patch_embed"] = x.clone()
    x = x + p["sam.pos_embed"]
    taps["sam_embed"] = x.clone()

    for i in range(cfg["sam_blocks"]):
        pre = f"sam.blocks.{i}."
        tag = f"sam_b{i}"
        b_in, h0, w0, _ = x.shape
        shortcut = x
        y = layer_norm(x, p[pre + "norm1.weight"], p[pre + "norm1.bias"], EPS_SAM)
        taps[f"{tag}_norm1"] = y.clone()
        wh, ww = cfg["sam_windows"][i]
        windowed = wh > 0
        pad_hw = None
        keep = None
        if windowed:
            y, pad_hw = window_partition(y, wh, ww)
            taps[f"{tag}_windows"] = y.clone()
            # `keep` is 1 on real positions and 0 on zero-padded ones: the same
            # partition applied to an all-ones grid.
            keep, _ = window_partition(torch.ones((b_in, h0, w0, 1)), wh, ww)
            taps[f"{tag}_keep"] = keep.clone()
            # Round-trip: unpartition(partition(x)) must be x, crop included.
            probe = torch.arange(b_in * h0 * w0 * c, dtype=torch.float32).reshape(
                b_in, h0, w0, c)
            parts, php = window_partition(probe, wh, ww)
            assert torch.equal(window_unpartition(parts, wh, ww, php, (h0, w0)), probe), \
                f"{tag}: window partition/unpartition is not a round trip"
            findings[f"{tag}_window"] = {
                "window": [wh, ww], "grid": [h0, w0], "padded": list(pad_hw),
                "windows_per_image": (pad_hw[0] // wh) * (pad_hw[1] // ww),
                "pad_positions_per_image": int((keep == 0).sum()) // b_in,
                "round_trip_verified": True,
            }
        b, hh, wwd, _ = y.shape
        n = hh * wwd
        qkv = linear(y.reshape(b, n, c), p[pre + "attn.qkv.weight"],
                     p[pre + "attn.qkv.bias"])
        taps[f"{tag}_qkv"] = qkv.clone()
        if windowed:
            # The pad is applied AFTER norm1, so a pad position's input is
            # exactly 0 and its qkv is exactly the qkv BIAS -- measured, not
            # assumed, because "pad tokens are inert" is the natural wrong guess
            # and it changes every window's softmax.
            pad_mask = keep.reshape(b, n) == 0
            assert pad_mask.any(), f"{tag}: window divides the grid, no pad to check"
            pad_gap = (qkv[pad_mask] - p[pre + "attn.qkv.bias"]).abs().max().item()
            assert pad_gap == 0.0, f"{tag}: pad qkv is not the bias ({pad_gap:.3e})"
            findings[f"{tag}_window"]["pad_qkv_equals_bias_max_abs"] = pad_gap
        q, k, v = (qkv.reshape(b, n, 3, heads, hd).permute(2, 0, 3, 1, 4)
                   .reshape(3, b * heads, n, hd).unbind(0))

        rh, info_h = get_rel_pos(hh, hh, p[pre + "attn.rel_pos_h"])
        rw, info_w = get_rel_pos(wwd, wwd, p[pre + "attn.rel_pos_w"])
        taps[f"{tag}_Rh"] = rh.clone()
        taps[f"{tag}_Rw"] = rw.clone()
        findings[f"{tag}_rel_pos_h"] = info_h
        findings[f"{tag}_rel_pos_w"] = info_w

        # SELF-CHECK 1: the bias term, two independent ways.
        bias = rel_pos_bias_einsum(q, rh, rw, (hh, wwd), (hh, wwd))
        bias_loops = rel_pos_bias_loops(q, rh, rw, (hh, wwd), (hh, wwd))
        gap = (bias - bias_loops).abs().max().item()
        assert torch.allclose(bias, bias_loops, atol=1e-5, rtol=0.0), \
            f"{tag}: rel-pos einsum vs explicit loops disagree by {gap:.3e}"
        findings[f"{tag}_rel_pos_two_ways_max_abs"] = gap
        taps[f"{tag}_relpos"] = bias.clone()
        taps[f"{tag}_relpos_loops"] = bias_loops.clone()

        # The score is a SCALED q.k^T, but the rel-pos term uses the UNSCALED q.
        scores = (q * (hd ** -0.5)) @ k.transpose(-2, -1) + bias
        probs = torch.softmax(scores, dim=-1)
        taps[f"{tag}_probs"] = probs.clone()
        ctx = (probs @ v).view(b, heads, hh, wwd, hd).permute(0, 2, 3, 1, 4).reshape(
            b, hh, wwd, c)
        taps[f"{tag}_ctx"] = ctx.clone()
        y = linear(ctx, p[pre + "attn.proj.weight"], p[pre + "attn.proj.bias"])
        if windowed:
            y = window_unpartition(y, wh, ww, pad_hw, (h0, w0))
        taps[f"{tag}_attn_out"] = y.clone()
        x = shortcut + y
        taps[f"{tag}_attn_res"] = x.clone()
        y = layer_norm(x, p[pre + "norm2.weight"], p[pre + "norm2.bias"], EPS_SAM)
        taps[f"{tag}_norm2"] = y.clone()
        # SAM's ViT MLP is nn.GELU, i.e. the EXACT erf form, not the tanh
        # approximation brain's gelu.wgsl computes for T5/CLIP-style stacks.
        hidden = F.gelu(linear(y, p[pre + "mlp.lin1.weight"], p[pre + "mlp.lin1.bias"]))
        taps[f"{tag}_mlp_hidden"] = hidden.clone()
        x = x + linear(hidden, p[pre + "mlp.lin2.weight"], p[pre + "mlp.lin2.bias"])
        taps[f"{tag}_out"] = x.clone()
    return x


def compressor_forward(x, p, taps):
    """[B, H, W, C] SAM output -> ([B, c_out, H', W'], [B, H' * W', c_out])."""
    y = x.permute(0, 3, 1, 2).contiguous()
    y = F.conv2d(y, p["compressor.neck.0.weight"])
    taps["neck_conv1"] = y.clone()
    y = layer_norm_2d(y, p["compressor.neck.1.weight"], p["compressor.neck.1.bias"])
    taps["neck_ln1"] = y.clone()
    y = F.conv2d(y, p["compressor.neck.2.weight"], padding=1)
    taps["neck_conv2"] = y.clone()
    y = layer_norm_2d(y, p["compressor.neck.3.weight"], p["compressor.neck.3.bias"])
    taps["neck_ln2"] = y.clone()
    y = F.conv2d(y, p["compressor.net_2.weight"], stride=2, padding=1)
    taps["comp_net2"] = y.clone()
    y = F.conv2d(y, p["compressor.net_3.weight"], stride=2, padding=1)
    taps["compressor_out"] = y.clone()
    flat = y.flatten(2).transpose(1, 2).contiguous()
    taps["compressor_flat"] = flat.clone()
    return y, flat


def clip_pos_embed(p, cfg, taps):
    """The learned table, resampled from the native square grid onto this run's.

    `interpolate_pos_encoding`: the class-token row never belongs to the patch
    grid, so it is lifted out, the patch rows go through the bicubic resample and
    the class row is put back at index 0. Height UPsamples and width DOWNsamples
    here, in one call, so a transposed `(h, w)` cannot survive.
    """
    w = cfg["clip_width"]
    native = cfg["clip_native_grid"]
    gh, gw = cfg["comp_grid_h"], cfg["comp_grid_w"]
    table = p["clip.position_embedding"].reshape(-1, w)
    cls_row = table[:1]
    grid = table[1:].reshape(1, native, native, w).permute(0, 3, 1, 2)
    resized = F.interpolate(grid, size=(gh, gw), mode="bicubic", align_corners=False)
    patch_rows = resized.permute(0, 2, 3, 1).reshape(gh * gw, w)
    full = torch.cat([cls_row, patch_rows], dim=0)
    taps["clip_pos_full"] = full.clone()
    return full.reshape(1, gh * gw + 1, w)


def clip_forward(patch_tokens, p, cfg, taps):
    """CLIP-L with its patch embed BYPASSED: patch tokens arrive pre-computed."""
    b, n, w = patch_tokens.shape
    assert w == cfg["clip_width"], (w, cfg["clip_width"])
    cls = p["clip.class_embedding"].reshape(1, 1, w).expand(b, 1, w)
    x = torch.cat([cls, patch_tokens], dim=1)
    taps["clip_cat"] = x.clone()
    x = x + clip_pos_embed(p, cfg, taps)
    taps["clip_tokens"] = x.clone()
    # The real mmproj carries `pre_ln` and NO post-LN: the tower's input is
    # normalised once and its output is the last block's hidden state verbatim.
    x = layer_norm(x, p["clip.pre_norm.weight"], p["clip.pre_norm.bias"], EPS_CLIP)
    taps["clip_pre_ln"] = x.clone()
    heads, hd = cfg["clip_heads"], cfg["clip_head_dim"]
    t = n + 1
    for i in range(cfg["clip_blocks"]):
        pre = f"clip.blocks.{i}."
        tag = f"clip_b{i}"
        shortcut = x
        y = layer_norm(x, p[pre + "norm1.weight"], p[pre + "norm1.bias"], EPS_CLIP)
        taps[f"{tag}_norm1"] = y.clone()
        qkv = []
        for proj in ["q_proj", "k_proj", "v_proj"]:
            z = linear(y, p[pre + f"attn.{proj}.weight"], p[pre + f"attn.{proj}.bias"])
            taps[f"{tag}_{proj[0]}"] = z.clone()
            qkv.append(z.reshape(b, t, heads, hd).transpose(1, 2).reshape(b * heads, t, hd))
        probs, ctx = softmax_attention(qkv[0], qkv[1], qkv[2], hd ** -0.5)
        taps[f"{tag}_probs"] = probs.clone()
        ctx = ctx.reshape(b, heads, t, hd).transpose(1, 2).reshape(b, t, w)
        taps[f"{tag}_ctx"] = ctx.clone()
        y = linear(ctx, p[pre + "attn.out_proj.weight"], p[pre + "attn.out_proj.bias"])
        taps[f"{tag}_attn_out"] = y.clone()
        x = shortcut + y
        taps[f"{tag}_attn_res"] = x.clone()
        y = layer_norm(x, p[pre + "norm2.weight"], p[pre + "norm2.bias"], EPS_CLIP)
        taps[f"{tag}_norm2"] = y.clone()
        y = linear(y, p[pre + "mlp.fc1.weight"], p[pre + "mlp.fc1.bias"])
        taps[f"{tag}_fc1"] = y.clone()
        y = quick_gelu(y)
        taps[f"{tag}_act"] = y.clone()
        x = x + linear(y, p[pre + "mlp.fc2.weight"], p[pre + "mlp.fc2.bias"])
        taps[f"{tag}_out"] = x.clone()
    taps["clip_out"] = x.clone()
    return x


def rope_tables(seq, head_dim, base):
    """NEOX (half-split) RoPE tables: cos/sin of shape [seq, head_dim]."""
    half = head_dim // 2
    inv = base ** (-torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim)
    freqs = torch.outer(torch.arange(seq, dtype=torch.float32), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rope_apply(x, cos, sin):
    """x: [B, T, D] with T-major positions; NEOX half-split rotation."""
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rot * sin


def swiglu(x, gate_w, up_w, down_w):
    return linear(F.silu(linear(x, gate_w)) * linear(x, up_w), down_w)


def decoder_forward(inputs_embeds, p, cfg, taps, findings):
    """DeepSeek-V2-style causal decoder: plain MHA + NEOX RoPE, dense then MoE."""
    b, t, d = inputs_embeds.shape
    heads, hd = cfg["dec_heads"], cfg["dec_head_dim"]
    inner = heads * hd
    cos, sin = rope_tables(t, hd, cfg["rope_base"])
    taps["rope_cos"] = cos.clone()
    taps["rope_sin"] = sin.clone()
    mask = torch.full((t, t), float("-inf")).triu(1)
    x = inputs_embeds
    for i in range(cfg["dec_layers"]):
        pre = f"decoder.layers.{i}."
        tag = f"dec_l{i}"
        shortcut = x
        y = rms_norm(x, p[pre + "input_layernorm.weight"])
        taps[f"{tag}_norm1"] = y.clone()
        qkv = []
        for proj in ["q_proj", "k_proj", "v_proj"]:
            z = linear(y, p[pre + f"self_attn.{proj}.weight"])
            taps[f"{tag}_{proj[0]}"] = z.clone()
            qkv.append(z.reshape(b, t, heads, hd).transpose(1, 2).reshape(b * heads, t, hd))
        q = rope_apply(qkv[0], cos, sin)
        k = rope_apply(qkv[1], cos, sin)
        taps[f"{tag}_q_rope"] = q.clone()
        taps[f"{tag}_k_rope"] = k.clone()
        probs, ctx = softmax_attention(q, k, qkv[2], hd ** -0.5, mask)
        taps[f"{tag}_probs"] = probs.clone()
        ctx = ctx.reshape(b, heads, t, hd).transpose(1, 2).reshape(b, t, inner)
        taps[f"{tag}_ctx"] = ctx.clone()
        y = linear(ctx, p[pre + "self_attn.o_proj.weight"])
        taps[f"{tag}_attn_out"] = y.clone()
        x = shortcut + y
        taps[f"{tag}_attn_res"] = x.clone()
        y = rms_norm(x, p[pre + "post_attention_layernorm.weight"])
        taps[f"{tag}_norm2"] = y.clone()
        if i == 0:
            gate = linear(y, p[pre + "mlp.gate_proj.weight"])
            up = linear(y, p[pre + "mlp.up_proj.weight"])
            taps[f"{tag}_ffn_gate"] = gate.clone()
            taps[f"{tag}_ffn_up"] = up.clone()
            ffn = linear(F.silu(gate) * up, p[pre + "mlp.down_proj.weight"])
        else:
            ffn = moe_forward(y, p, pre, cfg, taps, findings)
        taps[f"{tag}_ffn_out"] = ffn.clone()
        x = x + ffn
        taps[f"{tag}_out"] = x.clone()
    x = rms_norm(x, p["decoder.norm.weight"])
    taps["decoder_final_norm"] = x.clone()
    logits = linear(x, p["decoder.lm_head.weight"])
    taps["logits"] = logits.clone()
    return logits


def moe_forward(x, p, pre, cfg, taps, findings):
    """Softmax router, top-k, RAW (un-renormalized) gates, plus shared experts.

    SELF-CHECK 2 lives here: the gate is computed both with and without top-k
    renormalization and the two are required to DIFFER, so the fixture cannot be
    a degenerate case in which `norm_topk_prob` is unobservable.
    """
    b, t, d = x.shape
    flat = x.reshape(-1, d)
    logits = linear(flat, p[pre + "mlp.gate.weight"])
    probs = torch.softmax(logits, dim=-1)
    top_val, top_idx = torch.topk(probs, cfg["top_k"], dim=-1)
    gate_raw = top_val
    gate_renorm = top_val / top_val.sum(-1, keepdim=True)
    gap = (gate_raw - gate_renorm).abs().max().item()
    assert gap > 1e-3, (
        f"router gate is degenerate: raw and renormalized top-{cfg['top_k']} "
        f"gates differ by only {gap:.3e}, so this fixture cannot pin "
        "norm_topk_prob=false. Change the router seed/scale."
    )
    routed = torch.zeros_like(flat)
    for tok in range(flat.shape[0]):
        for j in range(cfg["top_k"]):
            e = int(top_idx[tok, j])
            ex = pre + f"mlp.experts.{e}."
            routed[tok] += gate_raw[tok, j] * swiglu(
                flat[tok], p[ex + "gate_proj.weight"], p[ex + "up_proj.weight"],
                p[ex + "down_proj.weight"])
    # The shared experts are a single fused SwiGLU added UNWEIGHTED (there is no
    # shared-expert gate tensor in the real checkpoint).
    sh = pre + "mlp.shared_experts."
    shared = swiglu(flat, p[sh + "gate_proj.weight"], p[sh + "up_proj.weight"],
                    p[sh + "down_proj.weight"])
    out = routed + shared

    # Prove the layer output really used the RAW gate: rebuilding it from the
    # renormalized gate must NOT reproduce it.
    renorm_routed = torch.zeros_like(flat)
    for tok in range(flat.shape[0]):
        for j in range(cfg["top_k"]):
            e = int(top_idx[tok, j])
            ex = pre + f"mlp.experts.{e}."
            renorm_routed[tok] += gate_renorm[tok, j] * swiglu(
                flat[tok], p[ex + "gate_proj.weight"], p[ex + "up_proj.weight"],
                p[ex + "down_proj.weight"])
    out_renorm = renorm_routed + shared
    out_gap = (out - out_renorm).abs().max().item()
    assert out_gap > 1e-4, "renormalized variant is indistinguishable -- bad probe"

    shape = (b, t, -1)
    taps["moe_router_logits"] = logits.reshape(shape).clone()
    taps["moe_probs"] = probs.reshape(shape).clone()
    taps["moe_topk_idx"] = top_idx.reshape(shape).to(torch.int32).clone()
    taps["moe_gate_raw"] = gate_raw.reshape(shape).clone()
    taps["moe_gate_renorm"] = gate_renorm.reshape(shape).clone()
    taps["moe_routed_out"] = routed.reshape(b, t, d).clone()
    taps["moe_shared_out"] = shared.reshape(b, t, d).clone()
    taps["moe_out_renorm_variant"] = out_renorm.reshape(b, t, d).clone()
    findings["router"] = {
        "scoring": "plain softmax over ALL experts, then top-k",
        "norm_topk_prob": False,
        "routed_scaling_factor": 1.0,
        "gate_used": "moe_gate_raw (the UN-renormalized softmax probabilities of "
                     "the selected experts)",
        "raw_vs_renorm_max_abs_gate": gap,
        "raw_vs_renorm_max_abs_layer_out": out_gap,
        "shared_experts": f"{cfg['n_shared']} shared experts fused into ONE "
                          f"{cfg['n_shared'] * cfg['moe_ff']}-wide SwiGLU, added "
                          "UNWEIGHTED (no shared-expert gate tensor exists)",
        "gate_raw": [round(v, 6) for v in gate_raw.flatten().tolist()],
        "gate_renorm": [round(v, 6) for v in gate_renorm.flatten().tolist()],
        "topk_idx": top_idx.flatten().tolist(),
    }
    return out.reshape(b, t, d)


def dump_tiny(out_dir, manifest):
    """A random whole-pipeline DeepSeek-OCR at dims that break its coincidences.

    Needs no checkpoint at all: the weights are seeded random, so this fixture
    can be regenerated anywhere and gates the port before any weights exist.
    """
    torch.manual_seed(SEED)
    cfg = dict(TINY)
    # Derived dims, computed rather than asserted by hand: conv output extent is
    # floor((L + 2p - k) / s) + 1.
    def conv_out(length, k, s, pad):
        return (length + 2 * pad - k) // s + 1

    gh = conv_out(cfg["image_h"], cfg["patch"], cfg["patch"], 0)
    gw = conv_out(cfg["image_w"], cfg["patch"], cfg["patch"], 0)
    assert (gh, gw) == (cfg["sam_grid_h"], cfg["sam_grid_w"]), (gh, gw)
    ch = conv_out(conv_out(gh, 3, 2, 1), 3, 2, 1)
    cw = conv_out(conv_out(gw, 3, 2, 1), 3, 2, 1)
    cfg["comp_grid_h"], cfg["comp_grid_w"] = ch, cw
    n_patch = ch * cw
    cfg["clip_tokens"] = n_patch + 1
    assert n_patch >= 4, f"compressor collapsed to {ch}x{cw} tokens"

    # CLIP's own native (square) patch grid, and the learned table's row count.
    assert cfg["clip_image_size"] % cfg["clip_patch"] == 0
    native = cfg["clip_image_size"] // cfg["clip_patch"]
    cfg["clip_native_grid"] = native
    cfg["clip_n_positions"] = native * native + 1
    assert (native, native) != (ch, cw), "the position resample must not be the identity"

    # The inequalities the whole fixture exists for, asserted rather than eyeballed.
    assert cfg["clip_width"] != cfg["c_out"], "concat-order gate needs distinct widths"
    assert cfg["d_model"] != cfg["clip_width"] + cfg["c_out"]
    assert len({cfg["dec_heads"], cfg["dec_head_dim"], cfg["d_model"]}) == 3
    assert cfg["dec_heads"] * cfg["dec_head_dim"] == cfg["d_model"]
    assert cfg["clip_heads"] * cfg["clip_head_dim"] == cfg["clip_width"]
    assert cfg["sam_heads"] * cfg["sam_head_dim"] == cfg["sam_embed"]
    assert cfg["top_k"] < cfg["n_routed"] and cfg["n_shared"] >= 1
    assert cfg["moe_ff"] != cfg["dense_ff"]
    wh, ww = cfg["sam_windows"][0]
    assert gh % wh and gw % ww, "block 0's window must divide NEITHER extent"
    for extent in [gh, gw, wh, ww, ch, cw]:
        assert cfg["sam_head_dim"] != extent, f"sam_head_dim collides with {extent}"
    assert len({gh, gw}) == 2 and len({wh, ww}) == 2 and len({ch, cw}) == 2
    assert n_patch + 1 != cfg["clip_head_dim"] != cfg["clip_width"]
    assert cfg["seq"] != cfg["d_model"] and cfg["seq"] > cfg["image_pos"] + n_patch
    # The consumer's binding-alignment rule (see the TINY block's header): one
    # windowed span is `wh*ww` rows and brain slices `ctx` at `row0 * sam_embed`
    # floats, which must be 64-float (256 B) aligned.
    assert (wh * ww * cfg["sam_embed"]) % 64 == 0, (
        f"a {wh}x{ww} window at embed {cfg['sam_embed']} gives a span offset of "
        f"{wh * ww * cfg['sam_embed']} floats, which is not 64-float aligned and "
        "cannot be dispatched"
    )
    assert cfg["batch"] == 1, "the SAM tower is single-image; see the TINY block's header"

    taps = {}
    findings = {}
    p = build_params(cfg)

    b = cfg["batch"]
    g = torch.Generator().manual_seed(SEED + 1)
    image = torch.randn((b, cfg["in_chans"], cfg["image_h"], cfg["image_w"]),
                        generator=g)
    taps["image"] = image.clone()

    sam_out = sam_forward(image, p, cfg, taps, findings)
    print(f"sam out {list(sam_out.shape)}", flush=True)
    _, comp_flat = compressor_forward(sam_out, p, taps)
    print(f"compressor {list(taps['compressor_out'].shape)} -> "
          f"{list(comp_flat.shape)}", flush=True)

    # CLIP's own conv patch embed is BYPASSED: the compressor output is injected
    # as the patch tokens. At real scale c_out == clip_width == 1024 and that
    # injection is verbatim; here the two widths are deliberately different (that
    # inequality is what makes a concat-order swap detectable), so the bypass
    # carries an explicit widening matrix. `compressor_flat` -- the tensor the
    # concat's high half is built from -- is tapped on BOTH sides of it.
    patch_tokens = linear(comp_flat, p["clip.patch_bypass.weight"],
                          p["clip.patch_bypass.bias"])
    taps["clip_patch_tokens"] = patch_tokens.clone()
    clip_out = clip_forward(patch_tokens, p, cfg, taps)
    print(f"clip out {list(clip_out.shape)}", flush=True)

    clip_spatial = clip_out[:, 1:, :].contiguous()
    taps["clip_spatial"] = clip_spatial.clone()
    vision_concat = torch.cat([clip_spatial, comp_flat], dim=-1)
    taps["vision_concat"] = vision_concat.clone()
    # SELF-CHECK 3: which half is which, tapped and asserted.
    w = cfg["clip_width"]
    assert torch.equal(vision_concat[..., :w], clip_spatial)
    assert torch.equal(vision_concat[..., w:], comp_flat)
    assert vision_concat.shape[-1] == w + cfg["c_out"]
    findings["vision_concat"] = {
        "order": "concat([clip_spatial, compressor_flat], dim=-1)",
        "low_half": f"[0, {w}) == CLIP output with its cls token DROPPED",
        "high_half": f"[{w}, {w + cfg['c_out']}) == the compressor's own "
                     "(pre-CLIP) spatial features",
        "halves_verified_by_slice": True,
        "widths_distinct": True,
    }
    findings["clip_position_embedding"] = {
        "table": f"[1 + {native}*{native}, {cfg['clip_width']}] on the checkpoint's "
                 f"NATIVE square grid; row 0 is the class token",
        "resample": f"patch rows {native}x{native} -> {ch}x{cw} with "
                    "F.interpolate(mode='bicubic', align_corners=False) "
                    "(a = -0.75 cubic convolution, clamp-to-edge taps); the class "
                    "row is lifted out first and re-inserted at index 0",
        "exercises": "height UPsamples and width DOWNsamples in the same call",
        "pre_norm": "the tower applies LayerNorm to [cls; tokens] + pos BEFORE "
                    "block 0 (the mmproj's `v.pre_ln`) and has NO post-LayerNorm",
    }
    findings["patch_bypass"] = {
        "what": "CLIP's conv patch embed is never run; the compressor output is "
                "injected as its patch tokens (clip_patch_tokens), a learned cls "
                "token is prepended and the learned absolute position embedding, "
                "resampled onto this grid, is added",
        "adapter": "clip.patch_bypass is a widening Linear(c_out, clip_width) "
                   "that exists ONLY in this fixture, which forces "
                   f"c_out {cfg['c_out']} != clip_width {cfg['clip_width']}; at "
                   "real scale both are 1024 and the injection is verbatim",
        "concat_half_is_pre_bypass": "the concat's high half is compressor_flat "
                                     "(pre-bypass), NOT clip_patch_tokens",
    }
    projector_out = linear(vision_concat, p["projector.weight"], p["projector.bias"])
    taps["projector_out"] = projector_out.clone()
    print(f"projector out {list(projector_out.shape)}", flush=True)

    ids = torch.randint(1, cfg["vocab"], (b, cfg["seq"]), generator=g)
    lo = cfg["image_pos"]
    ids[:, lo:lo + n_patch] = IMAGE_TOKEN_ID
    assert int((ids == IMAGE_TOKEN_ID).sum()) == b * n_patch
    taps["input_ids"] = ids.to(torch.int32)
    embeds = p["decoder.embed_tokens.weight"][ids]
    taps["token_embed"] = embeds.clone()
    spliced = embeds.clone()
    spliced[:, lo:lo + n_patch, :] = projector_out
    taps["decoder_input"] = spliced.clone()
    logits = decoder_forward(spliced, p, cfg, taps, findings)
    print(f"logits {list(logits.shape)}", flush=True)

    findings["attention"] = {
        "sam": "scores = (q * head_dim^-0.5) @ k^T + decomposed rel-pos bias, "
               "where the rel-pos term uses the UNSCALED q",
        "clip": "scores = (q * head_dim^-0.5) @ k^T, no mask, no position bias "
                "(position is the learned absolute embedding added once)",
        "decoder": "scores = (q * head_dim^-0.5) @ k^T + causal mask; plain MHA "
                   "(num_kv_heads == num_heads), NEOX half-split RoPE on q and k, "
                   "base 10000, full head_dim rotated",
    }
    findings["activations"] = {
        "sam_mlp": "exact erf GELU (nn.GELU), NOT the tanh approximation",
        "clip_mlp": "quick_gelu: x * sigmoid(1.702 * x)",
        "decoder_mlp": "SwiGLU: down(silu(gate(x)) * up(x))",
    }
    findings["norms"] = {
        "sam": f"LayerNorm eps {EPS_SAM} (weight+bias), channels-last",
        "compressor": f"LayerNorm2d eps {EPS_SAM}, over the CHANNEL dim of NCHW",
        "clip": f"LayerNorm eps {EPS_CLIP} (weight+bias)",
        "decoder": f"RMSNorm eps {EPS_RMS}, gain only, no mean subtraction",
    }
    findings["biases"] = {
        "with_bias": "sam patch_embed / qkv / attn.proj / mlp, clip q,k,v,out and "
                     "fc1,fc2, the projector",
        "without_bias": "every compressor conv, every decoder projection, the "
                        "router, every expert, lm_head",
    }

    save(out_dir, "tiny/ckpt/model.safetensors", p.t, manifest, extra={
        "reference": "hand-written DeepSeek-OCR reference forward (randomly "
                     "initialised, no checkpoint involved)",
        "why": "dims break the real model's coincidences: clip_width == c_out == "
               "1024, head_dim == 64 == compressor channels, square grids and "
               "windows",
        "naming": "flat dot-separated attribute paths; see the script docstring",
    })
    save(out_dir, "tiny/golden.safetensors", taps, manifest, extra={
        "config": cfg,
        "run": {"batch": b, "seed": SEED, "image": list(image.shape),
                "seq_len": cfg["seq"], "image_token_id": IMAGE_TOKEN_ID,
                "image_token_start": lo, "image_tokens": n_patch},
        "findings": findings,
        "notes": {
            "layout": "SAM taps are [B, H, W, C] channels-last; compressor taps "
                      "are NCHW; CLIP/decoder taps are [B, T, C]",
            "attention_taps": "sam_b*/clip_b*/dec_l* probs and ctx are "
                              "head-major, i.e. batch dim B*heads with head h of "
                              "image n at index n*heads + h",
            "sam_b0_windows": "post-norm1, post-zero-pad window partition, "
                              "[B*nwin, wh, ww, C] -- pad rows are exactly 0, so "
                              "their qkv equals the qkv BIAS, not 0 (asserted "
                              "here, recorded as pad_qkv_equals_bias_max_abs)",
            "sam_b0_keep": "the same partition applied to an all-ones grid: 1 on "
                           "real positions, 0 on zero-padded ones",
            "sam_b*_ctx": "attention context BEFORE attn.proj; in a WINDOWED "
                          "block it is still [B*nwin, wh, ww, C] (the "
                          "unpartition happens after the projection), in a "
                          "global block it is [B, H, W, C]",
            "relpos": "sam_b{i}_relpos is the additive [q, k] bias actually used; "
                      "sam_b{i}_relpos_loops is the independent recomputation",
            "moe_out_renorm_variant": "the MoE layer output rebuilt from the "
                                      "RENORMALIZED gate -- dumped only as the "
                                      "negative control; the real output is in "
                                      "dec_l1_ffn_out",
            "moe_topk_idx": "expert indices, stored as exactly-representable f32",
        },
    })
    return cfg


def dump_real(gguf_path, mmproj_path, out_dir, manifest):
    """Real-checkpoint taps from the Q8_0 GGUF pair. NOT YET IMPLEMENTED.

    TODO: dequantize the GGUF pair with the same importer the Rust side uses,
    run the reference forward at each supported resolution mode, and write
    `real/preprocess.safetensors`, `real/vision.safetensors` and
    `real/decoder.safetensors` as documented at the top of this file. The tiny
    fixture above is deliberately independent of this path: it needs no weights,
    so it gates the port long before any checkpoint is downloaded.
    """
    raise NotImplementedError(
        "the real-checkpoint dump is not implemented yet; run without --gguf/"
        f"--mmproj to write the tiny fixture only (asked for {gguf_path}, "
        f"{mmproj_path} -> {out_dir}, {len(manifest['files'])} files so far)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", help="DeepSeek-OCR LM GGUF. Omit to write ONLY the "
                                   "tiny fixture, which needs no checkpoint. "
                                   "NOT YET IMPLEMENTED.")
    ap.add_argument("--mmproj", help="the matching mmproj (vision) GGUF. "
                                     "NOT YET IMPLEMENTED.")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"files": {}, "params": {"seed": SEED, "torch": torch.__version__}}

    # The tiny fixture is checkpoint-free, so it goes first and always runs: a
    # broken environment fails here in a second instead of after dequantizing
    # gigabytes of weights.
    manifest["params"]["tiny"] = dump_tiny(args.out, manifest)

    name = "manifest.json"
    if args.gguf or args.mmproj:
        assert args.gguf and args.mmproj, "--gguf and --mmproj go together"
        manifest["params"]["gguf"] = os.path.abspath(args.gguf)
        manifest["params"]["mmproj"] = os.path.abspath(args.mmproj)
        dump_real(args.gguf, args.mmproj, args.out, manifest)
    else:
        # Never clobber a full manifest with a tiny-only run.
        name = "manifest-tiny.json"
        print("no --gguf/--mmproj: wrote the tiny fixture only", flush=True)

    with open(os.path.join(args.out, name), "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
