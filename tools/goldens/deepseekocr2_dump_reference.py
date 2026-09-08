#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump a checkpoint-free tiny reference for DeepSeek-OCR-2's NEW vision tower.

DeepSeek-OCR-2 keeps v1's SAM ViT-B tower and v1's MoE decoder unchanged -
`crates/sam1` and `crates/deepseek2` already carry their own gradcheck/golden
coverage, and this dumper does not re-derive either. What is new, and what this
fixture pins, is everything between them: the "DeepEncoder V2" resampler --

  SAM output tokens ++ learned query bank
      -> N Qwen2-shaped GQA blocks under a PREFIX-LM mask
      -> keep only the query half
      -> one Linear projector
      -> gather every view's projected queries plus one learned separator
         into the final row sequence spliced into the decoder

-- described below. Two real architectural FACTS (not implementation choices)
are baked into the tiny config on purpose, because getting either backwards is
the port bug this fixture exists to catch:

  * the SAM neck's output width and the encoder's hidden width are the SAME
    number in the real checkpoint (no adapter sits between them - `net_3`'s
    896 output channels feed the Qwen2 tower's 896-wide residual stream
    directly), so this fixture uses one width for both rather than two
    coincidentally-equal-looking numbers;
  * a view's image-token count and its learned-query-bank size are the SAME
    number by construction (the neck downsamples a tile to exactly `n_query`
    tokens, and the query bank selected for that view has exactly `n_query`
    rows) - so the mask's prefix boundary `P` is always half of that view's
    sequence length, never an independent parameter to get wrong.

Everything else that COULD be confused with something else in the same kernel
is a different number: the two views' `n_query` differ, GQA's head count and
KV-head count give a non-trivial repeat-KV group size, and the local grid is
wider than it is tall so a row-major-vs-column-major tile gather cannot pass by
accident.

Files written under `--out` (default `testdata/deepseekocr2`):

  tiny/ckpt/model.safetensors  the seeded-random weights: one shared encoder
                             (`encoder.layer{i}.*`), its final shared norm
                             (`encoder.norm.weight`), the two query banks
                             (`query_bank.{local,global}`), the projector and
                             the separator - the contract a future
                             `deepseekocr2::import` matches its own tiny-mode
                             loader against.
  tiny/golden.safetensors    per-view query-concat input, per-layer attention
                             scores before/after the prefix-LM mask and after
                             softmax, each layer's output, the post-final-norm
                             sequence, the query-half slice, the projector
                             output, and the final gathered row sequence (all
                             local tiles, the global view, one separator).
  manifest-tiny.json         shapes + sha256 of that file, the tiny config
                             (doubling as the golden's `source.identity`,
                             since there is no checkpoint), and the two design
                             invariants above, asserted rather than assumed.

The prefix-LM mask itself is brain's own already-shipped
`attn_prefix_mask.wgsl` arithmetic, restated here as the independent numpy-ish
(torch, CPU, no autograd needed) reference this dumper is required to be:

    allow(i, j) = (i < P and j < P) or (j <= i)

i.e. image rows (index < P) attend to every image row and nothing else; query
rows (index >= P) attend causally over the whole sequence, which reaches every
image row (P <= i) plus every earlier query row. Row i, column j is masked
with a large negative additive term whenever `allow` is false.

Usage:
  python3 tools/goldens/deepseekocr2_dump_reference.py --out testdata/deepseekocr2
"""

import argparse
import hashlib
import math
import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from golden_source import source_block

SEED = 0
EPS_RMS = 1e-6
NEG = -1.0e9

# Tiny dims. See the module docstring for which equalities are REAL facts
# (kept equal on purpose) and which are kept apart on purpose.
#
#   hidden = sam_width: the no-adapter seam between SAM's neck and the encoder.
#   heads=6, kv_heads=2 -> repeat-KV group size 3 (neither 1 nor `heads`).
#   n_query differs per view (5 local, 8 global) and equals that view's own
#   image-token count, so each view's prefix boundary P is self-determined.
#   local grid 3 wide x 2 tall: row-major (not square) so a transposed tile
#   gather cannot look right by accident.
#   decoder_hidden (projector out / separator width) differs from hidden.
TINY = {
    "hidden": 24,
    "heads": 6,
    "kv_heads": 2,
    "head_dim": 4,
    "ff": 17,
    "layers": 2,
    "rope_theta": 1_000_000.0,
    "n_query_local": 5,
    "n_query_global": 8,
    "tiles_w": 3,
    "tiles_h": 2,
    "decoder_hidden": 15,
}


def save(out_dir, rel, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifest["files"][rel] = {
        "sha256": sha,
        "bytes": os.path.getsize(path),
        "dtype": "F32",
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
    }
    print(f"wrote {rel}: {len(tensors)} tensors, {os.path.getsize(path) / 1e6:.3f} MB", flush=True)


class Params:
    """Flat name -> tensor dict of seeded-random weights, shared across views
    and tiles (this IS the real model's contract: one query bank per n_query,
    one set of encoder weights, one projector, applied identically to every
    view)."""

    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(seed)
        self.t = {}

    def new(self, name, shape, scale=0.4):
        assert name not in self.t, f"duplicate parameter {name}"
        v = torch.randn(shape, generator=self.g) * scale
        self.t[name] = v
        return v

    def gain(self, name, n):
        return self.new(name, (n,), scale=0.25) + 1.0

    def __getitem__(self, name):
        return self.t[name]


def rms_norm(x, w, eps=EPS_RMS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def silu(x):
    return x * torch.sigmoid(x)


def rope_neox(x, theta, base_pos=0):
    """Half-split ("NEOX") rotary embedding: the first half of `head_dim`
    rotates against the second half, not adjacent-pair ("GPT-J") interleaving.
    `x` is `[T, head_dim]`; positions run `base_pos .. base_pos+T`."""
    t, d = x.shape
    half = d // 2
    pos = torch.arange(base_pos, base_pos + t, dtype=torch.float32)
    inv_freq = theta ** (-torch.arange(0, half, dtype=torch.float32) / half)
    ang = pos[:, None] * inv_freq[None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[:, :half], x[:, half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def prefix_lm_mask(t, prefix):
    """`allow(i,j) = (i<P and j<P) or (j<=i)`, as an additive `[T,T]` bias --
    brain's `attn_prefix_mask.wgsl` restated independently. 0 where allowed,
    `NEG` where not."""
    i = torch.arange(t)[:, None]
    j = torch.arange(t)[None, :]
    allow = ((i < prefix) & (j < prefix)) | (j <= i)
    return torch.where(allow, torch.zeros(t, t), torch.full((t, t), NEG))


def gqa_layer(p, prefix_, x, cfg, weight_prefix, tap, tap_prefix):
    """One prefix-LM GQA Qwen2-shaped block: RMSNorm -> qkv(+bias) -> RoPE on
    q/k -> repeat-interleave KV to `heads` -> masked softmax attention ->
    out-proj -> residual -> RMSNorm -> SwiGLU MLP -> residual.

    `weight_prefix` names the SHARED layer weights (one encoder, run against
    every view and every tile); `tap_prefix` names where this particular
    call's intermediates land, which differs per view/tile even though the
    weights read from `weight_prefix` do not."""
    t, h = x.shape
    heads, kv_heads, hd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    group = heads // kv_heads

    xn = rms_norm(x, p[f"{weight_prefix}.ln1.weight"])
    q = xn @ p[f"{weight_prefix}.attn_q.weight"].t() + p[f"{weight_prefix}.attn_q.bias"]
    k = xn @ p[f"{weight_prefix}.attn_k.weight"].t() + p[f"{weight_prefix}.attn_k.bias"]
    v = xn @ p[f"{weight_prefix}.attn_v.weight"].t() + p[f"{weight_prefix}.attn_v.bias"]
    q = q.view(t, heads, hd)
    k = k.view(t, kv_heads, hd)
    v = v.view(t, kv_heads, hd)

    mask = prefix_lm_mask(t, prefix_)
    ctx = torch.empty(t, heads, hd)
    scale = 1.0 / math.sqrt(hd)
    scores_pre_all, scores_post_all, probs_all = [], [], []
    for head in range(heads):
        kvh = head // group
        qh = rope_neox(q[:, head, :], cfg["rope_theta"])
        kh = rope_neox(k[:, kvh, :], cfg["rope_theta"])
        scores_pre = (qh @ kh.t()) * scale
        scores_post = scores_pre + mask
        probs = torch.softmax(scores_post, dim=-1)
        ctx[:, head, :] = probs @ v[:, kvh, :]
        scores_pre_all.append(scores_pre)
        scores_post_all.append(scores_post)
        probs_all.append(probs)
    tap[f"{tap_prefix}.scores_pre_mask"] = torch.stack(scores_pre_all)
    tap[f"{tap_prefix}.scores_post_mask"] = torch.stack(scores_post_all)
    tap[f"{tap_prefix}.probs"] = torch.stack(probs_all)

    attn_out = ctx.reshape(t, heads * hd) @ p[f"{weight_prefix}.attn_out.weight"].t()
    x = x + attn_out

    xn2 = rms_norm(x, p[f"{weight_prefix}.ln2.weight"])
    gate = silu(xn2 @ p[f"{weight_prefix}.ffn_gate.weight"].t())
    up = xn2 @ p[f"{weight_prefix}.ffn_up.weight"].t()
    mlp_out = (gate * up) @ p[f"{weight_prefix}.ffn_down.weight"].t()
    x = x + mlp_out

    tap[f"{tap_prefix}.out"] = x
    return x


def resample_view(p, cfg, n_query, sam_tokens, query_bank, tap, view_name):
    """SAM tokens ++ query bank -> `cfg["layers"]` GQA-prefix blocks (ONE
    shared encoder, whatever the view or tile) -> one final shared RMSNorm ->
    keep the query half -> project. Returns the projected `[n_query,
    decoder_hidden]` result; every intermediate is recorded under
    `view_name.*`.

    The final norm is a real, separately-confirmed tensor (`v.post_ln` in the
    real mmproj header - see the ledger's M1 entry), applied ONCE per view
    after the last block and before the slice, with the same shared weight
    every view and tile reads - not a per-view parameter, same as every
    other weight this fixture's `build()` allocates only once."""
    x = torch.cat([sam_tokens, query_bank], dim=0)
    tap[f"{view_name}.concat_in"] = x
    for layer in range(cfg["layers"]):
        x = gqa_layer(p, n_query, x, cfg, f"encoder.layer{layer}", tap, f"{view_name}.layer{layer}")
    x = rms_norm(x, p["encoder.norm.weight"])
    tap[f"{view_name}.post_norm"] = x
    query_half = x[n_query:, :]
    tap[f"{view_name}.query_slice"] = query_half
    proj = query_half @ p["projector.weight"].t() + p["projector.bias"]
    tap[f"{view_name}.projected"] = proj
    return proj


def build(cfg, seed):
    p = Params(seed)
    hidden, ff, dh = cfg["hidden"], cfg["ff"], cfg["decoder_hidden"]
    heads, kv_heads, hd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]

    # ONE encoder: the real model has no per-view, per-tile parameters at
    # all - every tile and both view kinds run through the same Qwen2-shaped
    # stack. Only the query bank (below) is chosen per view.
    for layer in range(cfg["layers"]):
        pfx = f"encoder.layer{layer}"
        p.gain(f"{pfx}.ln1.weight", hidden)
        p.new(f"{pfx}.attn_q.weight", (heads * hd, hidden))
        p.new(f"{pfx}.attn_q.bias", (heads * hd,))
        p.new(f"{pfx}.attn_k.weight", (kv_heads * hd, hidden))
        p.new(f"{pfx}.attn_k.bias", (kv_heads * hd,))
        p.new(f"{pfx}.attn_v.weight", (kv_heads * hd, hidden))
        p.new(f"{pfx}.attn_v.bias", (kv_heads * hd,))
        p.new(f"{pfx}.attn_out.weight", (hidden, heads * hd))
        p.gain(f"{pfx}.ln2.weight", hidden)
        p.new(f"{pfx}.ffn_gate.weight", (ff, hidden))
        p.new(f"{pfx}.ffn_up.weight", (ff, hidden))
        p.new(f"{pfx}.ffn_down.weight", (hidden, ff))
    p.gain("encoder.norm.weight", hidden)
    p.new("query_bank.local", (cfg["n_query_local"], hidden))
    p.new("query_bank.global", (cfg["n_query_global"], hidden))
    p.new("projector.weight", (dh, hidden))
    p.new("projector.bias", (dh,))
    p.new("view_separator", (dh,))

    tap = {}
    n_tiles = cfg["tiles_w"] * cfg["tiles_h"]
    # SAM's own output is out of scope here (crates/sam1's own gate covers
    # it); a fresh seeded tensor per tile/view stands in for "whatever SAM
    # produced", at the width the neck's net_3 actually emits (== hidden,
    # per the no-adapter fact this fixture pins).
    local_rows = []
    for tile in range(n_tiles):
        sam_local = torch.randn(cfg["n_query_local"], hidden, generator=p.g) * 0.4
        tap[f"sam.local.tile{tile}"] = sam_local
        proj = resample_view(p, cfg, cfg["n_query_local"], sam_local, p["query_bank.local"], tap, f"local.tile{tile}")
        local_rows.append(proj)
    sam_global = torch.randn(cfg["n_query_global"], hidden, generator=p.g) * 0.4
    tap["sam.global"] = sam_global
    global_row = resample_view(p, cfg, cfg["n_query_global"], sam_global, p["query_bank.global"], tap, "global")

    # Row-major over the tile grid: tile index advances width-first (a
    # transposed height/width gather is visible because tiles_w != tiles_h),
    # then the single global view, then one separator row.
    gathered = torch.cat(local_rows + [global_row, p["view_separator"][None, :]], dim=0)
    tap["gathered_rows"] = gathered

    return p.t, tap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="testdata/deepseekocr2")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    manifest = {"files": {}, "tiny": TINY}
    ckpt, tap = build(TINY, args.seed)

    # The two design invariants the module docstring names, asserted rather
    # than assumed: this fixture would be lying about the real architecture
    # if either broke.
    assert ckpt["projector.weight"].shape[1] == TINY["hidden"], "SAM/encoder seam must share one width"
    assert tap["sam.local.tile0"].shape[0] == TINY["n_query_local"], "a view's image-token count == its n_query"
    assert tap["sam.global"].shape[0] == TINY["n_query_global"]

    save(args.out, "tiny/ckpt/model.safetensors", ckpt, manifest)
    save(args.out, "tiny/golden.safetensors", tap, manifest)
    manifest["source"] = source_block(
        checkpoint=None,
        files=(),
        # Only the integer-valued dims are shape-determining in the sense
        # `source_block` enforces exactly (int equality); `rope_theta` is a
        # float and stays in `manifest["tiny"]` instead.
        identity={k: v for k, v in TINY.items() if isinstance(v, int) and not isinstance(v, bool)},
    )
    manifest["invariants"] = {
        "sam_width_equals_encoder_hidden": True,
        "n_query_equals_view_image_tokens": True,
        "row_gather_order": "local tiles (row-major w-then-h), then global, then one separator",
    }
    with open(os.path.join(args.out, "manifest-tiny.json"), "w") as f:
        import json
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote manifest-tiny.json", flush=True)


if __name__ == "__main__":
    main()
