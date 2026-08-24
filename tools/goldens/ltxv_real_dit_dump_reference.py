#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump LTX-2.5 video-only DiT REAL-WEIGHT reference goldens (Phase 4, step 3).

Sibling of `ltxv_dit_dump_reference.py`, same self-validation discipline
(porting.md §1), but where that dumper proves the OP SEQUENCE at tiny random
dims, this one proves PORT CORRECTNESS at REAL width (`inner_dim=4096`,
32 heads x 128) on a REDUCED depth (2 of the real checkpoint's 48
`transformer_blocks`), loaded from the ACTUAL Q8_0 GGUF this session
downloaded - not a synthetic or narrowed checkpoint.

## Reference-strategy design decision (see the task's own plan doc, Phase 4)

Run the OFFICIAL `ltx_core.model.transformer.model.LTXModel` on tensors
dequantized by THIS SCRIPT'S OWN independent Q8_0 decoder (below), not by
loading a pre-converted brain checkpoint and not by calling into brain's own
Rust code at all. That is what makes this a PORT-correctness gate separate
from `crates/ltxv/tests/gguf_quant_real.rs`'s QUANTIZATION-correctness gate:
this script's `deq_q8_0` and brain's `checkpoint::gguf::deq_q8_0` are two
independent implementations of the same 34-byte-block arithmetic, and
`gguf_quant_real.rs` already proves they agree byte-for-byte on real tensors
(see that file - it is the reason this script does not need to re-prove
dequant correctness itself). Given that, this script's reference weights and
brain's Rust-loaded weights ARE numerically identical, so any divergence this
dumper's golden catches in the Rust replay test is a PORT bug, not a
quantization artifact.

## Why LTXModel(num_layers=2) instead of hand-building a truncated block list

`LTXModel.__init__`'s `num_layers` constructor arg controls how many
`BasicAVTransformerBlock`s it allocates - building it with `num_layers=2`
allocates ONLY 2 real-width blocks (~1.3 GB fp32, see this port's own
`ltxv_bench.rs` module doc for the per-layer estimate), never anywhere near
the real 48-layer/~51 GB shape. So the "instantiate a truncated model
directly from reference source, bypassing the full LTXModel wrapper"
contingency the task's plan names is not needed here: `LTXModel` already
IS that truncated construction, for free, via its own public constructor
argument - no reference-internals surgery required.

## Why NO checkpoint weights are written to disk here (unlike the tiny dumper)

`ltxv_dit_dump_reference.py` dumps its own (tiny, random, ~0.84 MB) weights
alongside its taps, because the tiny Rust test has no other way to get them.
Here the weights are REAL - dequantizing even this reduced 2-layer subset is
~2.7 GB fp32 - so re-dumping them as a committed golden fixture would be
absurd. Instead: brain's Rust replay test
(`crates/ltxv/tests/dit_parity.rs::ltxv_real_dit_tiny_layers_matches_reference`)
reads the SAME 2-layer weight subset directly off the SAME real GGUF file
(via `checkpoint::gguf::MmapGguf`, gated on the file's presence exactly like
every other real-weight test in this crate) - so only the small INPUT
tensors and the captured ACTIVATION taps are written here, matching what a
"forward hooks during a real run, Rust test becomes a pure replay" golden
needs to hold (porting.md §1) when the weights themselves are a shared,
already-on-disk fixture rather than something this dumper owns.

## Self-validation inside the dumper (porting.md §1)

1. Fresh-module determinism: a SECOND `LTXModel(num_layers=2)` instance,
   loaded with the SAME real weight subset, run on the SAME inputs, must
   produce a BIT-IDENTICAL output (eval mode, no dropout in this op
   sequence - proves loading is deterministic and the forward has no hidden
   RNG).
2. Batch-independence: the same sample replicated to batch 2 produces
   identical per-row output (catches cross-batch leakage - same check the
   tiny dumper makes).
3. RoPE unit-rotation invariant: `cos^2 + sin^2 == 1` on the captured RoPE
   tables.
4. `strict=True` `load_state_dict` - the reference model's OWN parameter
   set for this config must equal EXACTLY the real tensor names this script
   reads (a missing OR an extra name is a hard error, not a silent partial
   load) - the same two-way-coverage discipline `crate::import::
   validate_manifest` enforces on the Rust side, checked independently here.

## Small-first (this task's own hard constraint)

Grid `(2, 2, 2)` -> 8 tokens, `context_len=6` - the SAME tiny token/context
count `ltxv_dit_dump_reference.py`'s `TINY_CONFIG`/`GRID` already use (this
task's own instruction: reuse the existing small-shape precedent rather than
inventing a new one). Real WIDTH (4096) but only 2 of 48 layers and 8
tokens - forward cost here is a handful of `[8,4096]x[4096,4096]` matmuls,
not the 48-layer/large-token-count shape a real generation run would need.

Usage:
  python tools/goldens/ltxv_real_dit_dump_reference.py \\
      --gguf ~/.local/share/brain/models/Lightricks/LTX-2.5/ltx-2.5-22b-distilled-transformer-Q8_0.gguf \\
      --out testdata/golden/ltxv/dit [--seed 1234] [--layers 2]
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

_REFERENCE_ROOT = Path(os.environ.get(
    "LTXV_REFERENCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "resources" / "ltxv" / "source")))
sys.path.insert(0, str(_REFERENCE_ROOT / "packages" / "ltx-core" / "src"))

import ltx_core.model.transformer.transformer_args as transformer_args_mod  # noqa: E402
from ltx_core.model.transformer.model import LTXModel, LTXModelType  # noqa: E402
from ltx_core.model.transformer.modality import Modality  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

# ---------------------------------------------------------------------------
# A minimal, from-spec GGUF v2/v3 header reader (independent of brain's own
# Rust parser AND of any `gguf`/`llama-cpp-python`-style package - this repo's
# python env carries neither). Only reads what this dumper needs: the KV map
# (to confirm `general.architecture`) and each named tensor's raw on-disk
# bytes - decode (dequant) is this file's own, see `deq_q8_0` below.
# ---------------------------------------------------------------------------

_GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _skip_value(f, vtype):
    if vtype == 8:
        _read_str(f)
    elif vtype == 9:
        (atype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _skip_value(f, atype)
    else:
        f.read(_GGUF_SCALAR_SIZES[vtype])


class GgufLite:
    """Header-only GGUF reader: KV map (string-valued keys resolved, every
    other value type skipped - this dumper only ever reads
    `general.architecture`) + a name -> (dims, ggml_type, absolute_byte_start,
    on_disk_byte_length) index, mmap-backed so no tensor data is read until
    [`self.raw`] is called for a specific name."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            magic = f.read(4)
            assert magic == b"GGUF", f"{path}: not a GGUF file (magic {magic!r})"
            (self.version,) = struct.unpack("<I", f.read(4))
            (tcount,) = struct.unpack("<Q", f.read(8))
            (kcount,) = struct.unpack("<Q", f.read(8))
            self.kv = {}
            for _ in range(kcount):
                key = _read_str(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                if vtype == 8:
                    self.kv[key] = _read_str(f)
                else:
                    _skip_value(f, vtype)
            self.index = {}
            for _ in range(tcount):
                name = _read_str(f)
                (ndims,) = struct.unpack("<I", f.read(4))
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
                (ty,) = struct.unpack("<I", f.read(4))
                (offset,) = struct.unpack("<Q", f.read(8))
                self.index[name] = (dims, ty, offset)
            pos = f.tell()
            alignment = 32
            pad = (-pos) % alignment
            self.data_start = pos + pad
        self.file_size = os.path.getsize(path)

    def shape_torch(self, name):
        """torch/brain shape order (GGUF `ne` reversed - see brain's own
        `checkpoint::gguf` module doc for the same convention)."""
        dims, _ty, _off = self.index[name]
        return list(reversed(dims))

    def raw(self, name):
        """This tensor's raw on-disk bytes, still quantized if applicable."""
        dims, ty, offset = self.index[name]
        numel = 1
        for d in dims:
            numel *= d
        nbytes = _tensor_nbytes(ty, numel)
        start = self.data_start + offset
        with open(self.path, "rb") as f:
            f.seek(start)
            return f.read(nbytes), ty, numel


# ggml_type geometry for the two types this dumper's tensor subset actually
# uses (F32 for the never-quantized tables, Q8_0 for the 2D projections) -
# see `crate::int8::is_never_quantized`'s doc for why the split is exactly
# that.
_BLOCK_GEOM = {0: (1, 4), 8: (32, 34)}


def _tensor_nbytes(ty, numel):
    be, bb = _BLOCK_GEOM[ty]
    n_blocks = numel // be + (1 if numel % be else 0)
    return n_blocks * bb


def deq_q8_0(raw: bytes, numel: int):
    """From-spec GGUF Q8_0 dequant: 34-byte blocks, `[f16 le scale][32 int8]`,
    `value = int8 * scale` - independently re-derived from the format
    definition, the SAME algorithm `crates/ltxv/tests/gguf_quant_real.rs`'s
    `independent_deq_q8_0` implements in Rust (two independent
    implementations of the same arithmetic; that Rust test is what proves
    THIS function agrees with brain's own `checkpoint::gguf::deq_q8_0` on
    real tensors - this module doc's design-decision section explains why
    that means this script's weights and brain's are numerically identical)."""
    import numpy as np

    n_blocks = len(raw) // 34
    out = np.empty(n_blocks * 32, dtype=np.float32)
    for i in range(n_blocks):
        blk = raw[i * 34:(i + 1) * 34]
        scale = np.frombuffer(blk[0:2], dtype=np.float16)[0].astype(np.float32)
        qs = np.frombuffer(blk[2:34], dtype=np.int8).astype(np.float32)
        out[i * 32:(i + 1) * 32] = qs * scale
    return torch.from_numpy(out[:numel].copy())


def deq_f32(raw: bytes, numel: int):
    import numpy as np
    return torch.from_numpy(np.frombuffer(raw, dtype="<f4")[:numel].copy())


def load_real_tensor(g: GgufLite, name: str) -> torch.Tensor:
    raw, ty, numel = g.raw(name)
    if ty == 8:
        flat = deq_q8_0(raw, numel)
    elif ty == 0:
        flat = deq_f32(raw, numel)
    else:
        raise ValueError(f"{name}: unexpected ggml type {ty} (this dumper's tensor subset should be F32/Q8_0 only)")
    return flat.reshape(g.shape_torch(name))


# ---------------------------------------------------------------------------
# The real LTX-2.5 22B video-stream config (crate::config::LtxDitConfig::
# ltx25_22b's own values, transcribed identically - both sides read the same
# GGUF `config` KV, this is the cross-check).
# ---------------------------------------------------------------------------

REAL_CONFIG = dict(
    model_type=LTXModelType.VideoOnly,
    num_attention_heads=32,
    attention_head_dim=128,          # inner_dim = 4096
    in_channels=128,
    out_channels=128,
    cross_attention_dim=4096,        # == inner_dim, no caption_projection (see below)
    norm_eps=1e-6,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    apply_gated_attention=True,       # real LTX-2.5 value
    caption_projection=None,          # caption_proj_before_connector=true, see ltxv_dit_dump_reference.py's doc
    cross_attention_adaln=True,
    use_prompt_adaln_single=False,
    ff_bias=False,
    use_keyframes_abs_pos_embedding=True,
)

GRID = (2, 2, 2)   # SAME tiny grid as ltxv_dit_dump_reference.py's TINY_CONFIG -> T = 8 tokens
CONTEXT_LEN = 6    # SAME as the tiny dumper


def real_tensor_names(num_layers: int):
    """Every tensor name this reduced-depth config needs - mirrors
    `crate::dit::dit_tensor_manifest` exactly (both are transcribed from the
    same real checkpoint header), so the Rust replay test reads the IDENTICAL
    name set from the SAME file."""
    names = [
        "patchify_proj.weight", "patchify_proj.bias",
        "adaln_single.emb.timestep_embedder.linear_1.weight", "adaln_single.emb.timestep_embedder.linear_1.bias",
        "adaln_single.emb.timestep_embedder.linear_2.weight", "adaln_single.emb.timestep_embedder.linear_2.bias",
        "adaln_single.linear.weight", "adaln_single.linear.bias",
        "scale_shift_table",
        "proj_out.weight", "proj_out.bias",
        "keyframes_abs_pos_embedding",
    ]
    for l in range(num_layers):
        p = f"transformer_blocks.{l}"
        for attn in ("attn1", "attn2"):
            for proj in ("to_q", "to_k", "to_v", "to_out.0"):
                names.append(f"{p}.{attn}.{proj}.weight")
                names.append(f"{p}.{attn}.{proj}.bias")
            names += [f"{p}.{attn}.q_norm.weight", f"{p}.{attn}.k_norm.weight",
                      f"{p}.{attn}.to_gate_logits.weight", f"{p}.{attn}.to_gate_logits.bias"]
        names += [f"{p}.ff.net.0.proj.weight", f"{p}.ff.net.2.weight",
                  f"{p}.scale_shift_table", f"{p}.prompt_scale_shift_table"]
    return names


def build_real_model(g: GgufLite, num_layers: int, seed: int):
    """Build `LTXModel(num_layers=num_layers)` at real width and load it with
    the REAL, dequantized weight subset - `strict=True`, so a name mismatch
    between this script's `real_tensor_names` and the reference class's own
    parameter set is a hard error (this file's own self-validation #4)."""
    cfg = dict(REAL_CONFIG, num_layers=num_layers)
    torch.manual_seed(seed)  # irrelevant to the loaded weights, kept for any torch-internal init order
    model = LTXModel(**cfg)
    sd = {name: load_real_tensor(g, name) for name in real_tensor_names(num_layers)}
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model.eval().requires_grad_(False)
    return model


def det_video_modality(seed, grid, context_len, inner_dim, in_channels, keyframe_token=0):
    """Byte-for-byte the same construction as `ltxv_dit_dump_reference.py`'s
    `det_video_modality` (see that function's doc) - reused here rather than
    imported so this file stays a fully self-contained sibling."""
    g = torch.Generator().manual_seed(seed)
    f, h, w = grid
    t = f * h * w
    b = 1

    latent = torch.randn(b, t, in_channels, generator=g)
    sigma = torch.tensor([0.7])
    denoise_mask = torch.ones(b, t, 1)
    timesteps = denoise_mask * sigma.view(-1, 1, 1)

    grid_coords = torch.meshgrid(torch.arange(f), torch.arange(h), torch.arange(w), indexing="ij")
    starts = torch.stack(grid_coords, dim=0).to(torch.float32)
    ends = starts + 1.0
    bounds = torch.stack([starts, ends], dim=-1)
    positions = einops.repeat(bounds, "c f h w bounds -> bs c (f h w) bounds", bs=b)

    context = 0.5 * torch.randn(b, context_len, inner_dim, generator=g)

    keyframes_mask = torch.zeros(b, t, 1)
    keyframes_mask[:, keyframe_token, :] = 1.0

    return Modality(latent=latent, sigma=sigma, timesteps=timesteps, positions=positions,
                     context=context, keyframes_mask=keyframes_mask)


class Taps:
    def __init__(self):
        self.acc, self.handles = {}, []

    def watch(self, name, module, pick=lambda o: o):
        def hook(_m, _i, o):
            self.acc[name] = _clone(pick(o))
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _clone(x):
    if isinstance(x, tuple):
        return tuple(_clone(v) for v in x)
    return x.detach().clone()


def run_with_taps(model, video, num_layers):
    captured_rope = {}
    orig_precompute = transformer_args_mod.precompute_freqs_cis

    def _capture_rope(*args, **kwargs):
        result = orig_precompute(*args, **kwargs)
        captured_rope["pe"] = result
        return result

    transformer_args_mod.precompute_freqs_cis = _capture_rope
    try:
        taps = Taps()
        taps.watch("adaln_single", model.adaln_single)
        for i in range(num_layers):
            block = model.transformer_blocks[i]
            taps.watch(f"block.{i}", block, pick=lambda o: o[0].x)
        b0 = model.transformer_blocks[0]
        taps.watch("b0.attn1", b0.attn1)
        taps.watch("b0.attn2", b0.attn2)
        taps.watch("b0.ff", b0.ff)

        out_v, out_a = model(video=video, audio=None, perturbations=None)
        assert out_a is None
    finally:
        transformer_args_mod.precompute_freqs_cis = orig_precompute
    taps.close()

    return out_v, dict(taps.acc), captured_rope["pe"]


def agree(label, a, b, tol=1e-6):
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    cos = F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item() if a.numel() > 1 else 1.0
    print(f"  self-validate {label}: max_abs {d:.3e} / scale {scale:.3g} = {rel:.2e} "
          f"(tol {tol:g}), cosine {cos:.10f}", flush=True)
    assert rel <= tol, f"{label}: disagree by {rel:.3e} relative"


def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    total = sum(v.numel() for v in tensors.values()) * 4 / 1e6
    print(f"wrote {name}: {len(tensors)} tensors, {total:.2f} MB", flush=True)
    manifest.setdefault("files", {})[name] = {k: list(v.shape) for k, v in tensors.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="path to the real ltx-2.5-22b-distilled-transformer-Q8_0.gguf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layers", type=int, default=2, help="reduced depth - see this file's module doc for why 2, not 48")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    print(f"opening {args.gguf} ...", flush=True)
    g = GgufLite(args.gguf)
    arch = g.kv.get("general.architecture")
    assert arch == "ltxv", f"expected general.architecture=ltxv, got {arch!r}"
    print(f"  GGUF v{g.version}, {len(g.index)} tensors, {g.file_size / 1e9:.2f} GB, architecture={arch}", flush=True)

    model = build_real_model(g, args.layers, args.seed)
    print(f"real-width model built: inner_dim={model.inner_dim}, layers={args.layers} (of the real checkpoint's 48)", flush=True)

    video = det_video_modality(args.seed, GRID, CONTEXT_LEN, model.inner_dim, REAL_CONFIG["in_channels"])
    f, h, w = GRID
    tokens = f * h * w
    print(f"grid={GRID} -> {tokens} tokens, context_len={CONTEXT_LEN} (same small shape as the tiny-config dumper)", flush=True)

    out_v, taps, (rope_cos, rope_sin) = run_with_taps(model, video, args.layers)
    assert not out_v.isnan().any(), "output contains NaN"

    # ---- self-validation 1: fresh module + fresh real-weight load, bit-identical
    model2 = build_real_model(g, args.layers, args.seed)
    out_v2, _, _ = run_with_taps(model2, video, args.layers)
    agree("fresh-instantiation (real weights) output", out_v2, out_v, tol=0.0)
    del model2

    # ---- self-validation 2: batch independence -------------------------------
    video_b2 = Modality(
        latent=video.latent.repeat(2, 1, 1), sigma=video.sigma.repeat(2),
        timesteps=video.timesteps.repeat(2, 1, 1), positions=video.positions.repeat(2, 1, 1, 1),
        context=video.context.repeat(2, 1, 1), keyframes_mask=video.keyframes_mask.repeat(2, 1, 1))
    out_v_b2, _, _ = run_with_taps(model, video_b2, args.layers)
    agree("batch-independence row 0", out_v_b2[0], out_v[0], tol=1e-5)
    agree("batch-independence row 1", out_v_b2[1], out_v[0], tol=1e-5)

    # ---- self-validation 3: RoPE unit-rotation invariant ----------------------
    unit = rope_cos.double() ** 2 + rope_sin.double() ** 2
    max_dev = (unit - 1.0).abs().max().item()
    print(f"  self-validate RoPE cos^2+sin^2==1: max deviation {max_dev:.3e}", flush=True)
    assert max_dev < 1e-5, f"RoPE tables are not unit rotations (max dev {max_dev:.3e})"

    tensors = {
        "latent": video.latent[0],
        "context": video.context[0],
        "timesteps": video.timesteps[0],
        "positions": video.positions[0],
        "keyframes_mask": video.keyframes_mask[0],
        "rope_cos": rope_cos[0],
        "rope_sin": rope_sin[0],
        "adaln_table": taps["adaln_single"][0],
        "embedded_timestep": taps["adaln_single"][1],
        "b0_attn1_out": taps["b0.attn1"][0],
        "b0_attn2_out": taps["b0.attn2"][0],
        "b0_ff_out": taps["b0.ff"][0],
        "out": out_v[0],
    }
    for i in range(args.layers):
        tensors[f"block.{i}.out"] = taps[f"block.{i}"][0]

    manifest = {
        "run": {
            "seed": args.seed, "grid": list(GRID), "tokens": tokens, "context_len": CONTEXT_LEN,
            "layers": args.layers, "real_config": {k: (v.value if hasattr(v, "value") else v)
                                                     for k, v in REAL_CONFIG.items() if k not in ("caption_projection",)},
        },
        "source_checkpoint": {
            "path_basename": os.path.basename(args.gguf), "gguf_version": g.version,
            "total_tensor_count": len(g.index), "file_size_bytes": g.file_size,
            "architecture": arch, "tensor_names_used": real_tensor_names(args.layers),
        },
        "versions": {"torch": torch.__version__, "einops": einops.__version__, "python": sys.version.split()[0]},
    }
    save(args.out, "dit_real_tiny.safetensors", tensors, manifest)

    # Real 22B weights at REDUCED depth, so `num_layers` here is `args.layers`
    # (this dump's own truncation), not the checkpoint's - and recording it is
    # the point: a reader pairing this golden with a full-depth run is exactly
    # the mismatch to catch. This manifest already carries a `source_checkpoint`
    # block of GGUF forensics; `source` is the ENFORCED half that
    # brain_testutil::golden::Source compares, which that block is not.
    manifest["source"] = source_block(
        checkpoint="Lightricks/LTX-2.5",
        files=[args.gguf],
        hash_files=False,
        identity={
            "num_attention_heads": REAL_CONFIG["num_attention_heads"],
            "attention_head_dim": REAL_CONFIG["attention_head_dim"],
            "in_channels": REAL_CONFIG["in_channels"],
            "out_channels": REAL_CONFIG["out_channels"],
            "cross_attention_dim": REAL_CONFIG["cross_attention_dim"],
            "num_layers": int(args.layers),
        },
    )

    with open(os.path.join(args.out, "manifest_real.json"), "w") as f_:
        json.dump(manifest, f_, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}/manifest_real.json", flush=True)


if __name__ == "__main__":
    main()
