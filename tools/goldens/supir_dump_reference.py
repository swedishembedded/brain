#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump SUPIR reference goldens for brain's `crates/supir` parity ladder.

SUPIR ("Scaling Up to Excellence") is not a `diffusers`-native model - it
ships its own from-scratch framework (`sgm/`, bundled inside the upstream
`Fanghua-Yu/SUPIR` repo) with no pip package. This dumper drives that upstream
Python source directly (via `--supir-src`, a local clone - see below), builds
the REAL `SUPIRModel` (`sgm.models.diffusion.DiffusionEngine` subclass) from
the real checkpoints, and taps a REAL forward pass with forward hooks and a
handful of narrowly-scoped class-level monkeypatches on the sampler/guider (to
reach per-step internals `RestoreEDMSampler`/`LinearCFG` do not otherwise
expose), so no convention - the trunk's hint-injection point, the adaptors'
input/output framing, the per-step control-scale/CFG ramps, the sigma
snapping - is re-derived by hand; the Rust parity test is a pure replay.

Verified upstream commit (pin exactly, see `--supir-src`):
    Fanghua-Yu/SUPIR @ bda91af2000042f8bedfec8897d92917e67c1d88

This dumper implements against an architecture spec already verified against
upstream source and the real checkpoint headers (weight manifest, the
GLVControl trunk's ten hidden-state outputs, the twelve ZeroSFT/ZeroCrossAttn
adaptors, the RestoreEDMSampler math) - not re-derived here.

Output layout, written under `--out`:
  stages.safetensors    every tap: VAE-stage taps (_z's moments/mode/sample,
                        x_stage1, z_stage1 and its mode for comparison), both
                        CLIP towers' per-tower AND merged crossattn/vector
                        conditioning, the 10 GLVControl trunk outputs (`hs`),
                        the input(s) AND output of all 12 `project_modules`,
                        the `LightGLVUNet` forward's raw final output (pre
                        EDM c_skip/c_out), the sigma schedule (sampler path +
                        an independent hand-computed path), the per-step
                        latent/sigma/ramped-control-scale/ramped-CFG-scale
                        trajectory, and the final VAE decode.
  manifest.json         every tensor's shape, sha256 per file, the run
                        parameters, and the `source` provenance block
                        (`golden_source.source_block`) naming the exact
                        checkpoints and the architecture identity that fixes
                        every dumped shape.

Everything is CPU + fp32 with fixed seeds. Every tensor is stored as f32
(brain's safetensors reader is F32/F16/BF16-only) - the upstream SDXL base
checkpoint ships in fp16; loading it into an fp32-initialized model upcasts
losslessly via `Tensor.copy_`'s implicit cast, so no precision is lost, it is
simply not free-lunch-narrow like the SUPIR-v0Q checkpoint (which IS fp32 on
disk already).

Scope, stated honestly (same convention as `controlnet_dump_reference.py` /
`sdxlunet_dump_reference.py`): dumped at a SMALL latent (`--latent 32`, i.e. a
256x256 LQ image - the VAE's factor-8 downsample) and a SMALL step count
(`--steps 4`, not upstream's `edm_steps 50` CLI default) so the golden set and
the CPU run time both stay small. The graph is resolution- and
step-count-independent, so this gates the COMPOSITION (trunk -> adaptors ->
UNet -> sampler loop -> VAE), not native-resolution/native-step-count
behaviour.

Two deliberate departures from the upstream CLI's OWN defaults, so the golden
actually exercises the branches a real port must get right instead of the
branches that happen to be disabled by default:
  - `restoration_scale` (`s_stage1`/`restore_cfg`) is set to the YAML's `4.0`,
    NOT the CLI default's `-1` (OFF) - the restoration-guidance branch in
    `RestoreEDMSampler.sampler_step` (`denoised -= (denoised - x_center) *
    (sigma/sigma_max)**restore_cfg`) is exactly the kind of scalar-math branch
    a port can silently get backwards, so it is turned ON here and asserted
    against a hand-computed value below.
  - `use_linear_control_scale=True` with a nonzero `control_scale_start` - so
    the per-step control-scale RAMP is actually exercised (a constant
    `control_scale=1.0` throughout would make a ramp bug invisible).

A real finding worth flagging wherever this port's architecture spec is
ledgered: a documented pipeline formula reading
`z_stage1 = 0.13025 . quant_conv(encoder(x_stage1)).mode()` is WRONG. The REAL
upstream code (`SUPIR_model.py::SUPIRModel.encode_first_stage`, which calls
`AutoencoderKLInferenceWrapper.encode` -> `AutoencoderKL.encode(x).sample()`)
uses `.sample()`, NOT `.mode()` - `z_stage1` is a genuinely stochastic draw
(`mean + std * randn`), reproducible only via the fixed seed, not a
deterministic function of `x_stage1` alone. This dumper records BOTH the real
(sampled) `z_stage1` AND an independently-computed `z_stage1.mode` (mean-only,
recomputed from the same tapped moments) side by side, so the Rust side can
see the actual gap between the two conventions rather than inherit a silently
wrong assumption.

Requires a DEDICATED virtualenv, not this workspace's ambient Python. The
upstream `sgm` package needs `open-clip-torch==2.17.1` exactly
(`requirements.txt`); a newer `open_clip` (this environment's ambient one is
3.3.0) changes `open_clip.transformer`'s internal API in a way that crashes
deep inside the bigG text tower's attention with a shape mismatch, not an
import error - the version skew is silent until it runs. Build the venv once:

  python3 -m venv /path/to/supir-venv
  /path/to/supir-venv/bin/pip install \\
      open-clip-torch==2.17.1 torch torchvision omegaconf pytorch-lightning \\
      kornia einops transformers safetensors huggingface_hub scipy \\
      k-diffusion diffusers "protobuf<4"

(`protobuf<4` last, on purpose - `k-diffusion` pulls in `wandb`, which wants a
newer `protobuf` than `open-clip-torch` tolerates; the older pin is what this
crate's own dependency, not wandb's, actually needs.) Then run the dumper with
THAT venv's `python3`, not the ambient one:

Usage:
  /path/to/supir-venv/bin/python3 tools/goldens/supir_dump_reference.py \\
      --supir-src /path/to/cloned/Fanghua-Yu/SUPIR \\
      --sdxl      /path/to/sd_xl_base_1.0_0.9vae.safetensors \\
      --supir     /path/to/SUPIR-v0Q_fp32.safetensors \\
      --clip-l    /path/to/clip-vit-large-patch14/dir \\
      --clip-bigg /path/to/open_clip_model.safetensors \\
      --out testdata/supir [--latent 32] [--steps 4] [--s-churn 5.0]

`--s-churn` defaults to `5.0`, matching the committed `testdata/supir/`
golden bit-for-bit when omitted. Pass `--s-churn 0.0` (into a DIFFERENT
`--out` directory, never overwriting `testdata/supir/`) for a
forward-parity golden: with churn off, `gamma == 0` so `sigma_hat == sigma`
and every per-step tap is a deterministic function of the seed alone, with
no unrecoverable churn-noise draw standing between the dump and a Rust-side
forward replay - see `--s-churn`'s own `--help` text for why the default
golden cannot be used for that.
"""

import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SEED = 20260827
PROMPT = "a professional, detailed, high-quality photo of a street market"
LATENT_DEFAULT = 32
STEPS_DEFAULT = 4

# Sampling knobs. See the module docstring for why `restoration_scale` and
# `use_linear_control_scale` deliberately depart from the upstream CLI's own
# (branch-disabling) defaults.
CFG_SCALE = 7.5          # s_cfg, Quality preset (roadmap "Defaults")
CFG_SCALE_START = 4.0    # spt_linear_CFG, Quality preset
CONTROL_SCALE = 1.0      # s_stage2
CONTROL_SCALE_START = 0.0
RESTORATION_SCALE = 4.0  # s_stage1 / restore_cfg - ON (see docstring)
S_CHURN_DEFAULT = 5.0    # overridable via --s-churn; see --s-churn's own help
S_NOISE = 1.01


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(store, name, t):
    """Store a tensor as contiguous fp32 on the CPU."""
    if isinstance(t, (tuple, list)):
        t = t[0]
    store[name] = t.detach().to(torch.float32).contiguous().cpu().clone()


def save(store, out, rel, manifest):
    path = os.path.join(out, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(store, path)
    manifest["files"][rel] = {
        "sha256": sha256(path),
        "tensors": {k: list(v.shape) for k, v in sorted(store.items())},
    }
    print(f"  wrote {rel}: {len(store)} tensors, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)


# ---------------------------------------------------------------------------
# Streaming (one-tensor-at-a-time) safetensors loading.
#
# This machine has 30 GB total RAM and no discrete GPU; the SDXL base (3.47B
# params) and SUPIR-v0Q (1.33B params) checkpoints both landing fully in
# memory as Python dicts AT THE SAME TIME as the (already-allocated, fp32)
# model they are being copied into risks a transient peak north of what is
# available (`model.load_state_dict(huge_dict, strict=False)` keeps the WHOLE
# source dict resident for the duration of the call). `safe_open` lets each
# tensor be read and copied in isolation, so the transient cost is one tensor
# (tens of MB, not gigabytes) instead of the whole file.
# ---------------------------------------------------------------------------
def load_state_dict_streaming(model, path):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    targets = {**params, **buffers}
    seen = set()
    unexpected = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            dst = targets.get(key)
            if dst is None:
                unexpected.append(key)
            else:
                dst.data.copy_(tensor)
                seen.add(key)
            del tensor
    missing = sorted(set(targets.keys()) - seen)
    return missing, unexpected


def _patch_openclip_lazy_text_load():
    """Avoid ever fully materialising OpenCLIP bigG's VISION tower.

    `FrozenOpenCLIPEmbedder2.__init__` builds the model with BOTH towers via
    `open_clip.create_model_and_transforms(pretrained=<local file>)`, then
    does `del model.visual`. As written, that still means: (a) the full
    checkpoint (10.16 GB, both towers) is read into memory eagerly by
    `open_clip`'s own `load_state_dict` (`safetensors.torch.load_file`), and
    (b) the vision tower (~1.84B params, ~7.4 GB fp32) is allocated at random
    init before being freed. Both are dead weight for SUPIR, which only ever
    uses the text tower. This patches `open_clip.create_model_and_transforms`
    to build the model with NO pretrained weights (fast, no disk read), then
    stream in only the non-`visual.*` tensors via `safe_open` before deleting
    `.visual` exactly as upstream already does - the peak this call adds is
    one tensor at a time, not 10+ GB.
    """
    import open_clip

    orig = open_clip.create_model_and_transforms

    def patched(model_name, pretrained=None, device="cpu", **kwargs):
        is_local_file = isinstance(pretrained, str) and os.path.isfile(pretrained)
        if not is_local_file:
            return orig(model_name, pretrained=pretrained, device=device, **kwargs)
        model, pre_t, pre_v = orig(model_name, pretrained=None, device=device, **kwargs)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        targets = {**params, **buffers}
        with safe_open(pretrained, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("visual."):
                    continue
                dst = targets.get(key)
                if dst is None:
                    continue
                tensor = f.get_tensor(key)
                dst.data.copy_(tensor)
                del tensor
        return model, pre_t, pre_v

    open_clip.create_model_and_transforms = patched


def _patch_vae_attn_without_xformers():
    """Fall back the VAE's attention to native SDPA when xformers is absent.

    `sgm/modules/diffusionmodules/model.py::make_attn` (the VAE's attention
    factory) only downgrades `attn_type="vanilla-xformers"` to `"vanilla"` when
    BOTH torch < 2.0 AND xformers is unavailable; on torch >= 2.0 it leaves
    `attn_type` exactly as the config requested, with no availability check at
    all. The SUPIR VAE config (`options/SUPIR_v0.yaml`'s `first_stage_config.
    ddconfig.attn_type`) hardcodes `"vanilla-xformers"`, so on a modern-torch,
    no-xformers machine `MemoryEfficientAttnBlock.attention` unconditionally
    calls `xformers.ops.memory_efficient_attention` and raises `NameError` the
    first time the VAE runs. This is a genuine inconsistency with the SAME
    repo's `sgm/modules/attention.py` (the UNet's SpatialTransformer attention),
    which DOES check `XFORMERS_IS_AVAILABLE` unconditionally and degrades to
    native attention (that path already worked, printing "Falling back to
    native attention" during model construction above). This patches
    `make_attn` to apply that same availability check to the VAE path:
    `AttnBlock` ("vanilla") is mathematically the same single-head, no-mask,
    full self-attention as `MemoryEfficientAttnBlock` ("vanilla-xformers") -
    one uses `torch.nn.functional.scaled_dot_product_attention` directly, the
    other routes the identical math through `xformers.ops`, so downgrading
    changes no numerics, only the kernel.
    """
    from sgm.modules.diffusionmodules import model as dm_model

    orig_make_attn = dm_model.make_attn

    def patched_make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
        if attn_type == "vanilla-xformers" and not dm_model.XFORMERS_IS_AVAILABLE:
            attn_type = "vanilla"
        return orig_make_attn(in_channels, attn_type=attn_type, attn_kwargs=attn_kwargs)

    dm_model.make_attn = patched_make_attn


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(args):
    sys.path.insert(0, args.supir_src)

    # `sgm.modules.encoders.modules` does `from CKPT_PTH import
    # SDXL_CLIP1_PATH, SDXL_CLIP2_CKPT_PTH` at module import time - CKPT_PTH
    # is a plain root-level module in the upstream repo carrying
    # machine-specific paths. Rather than editing the clone on disk, inject an
    # in-memory module BEFORE anything imports it, pointed at our real local
    # checkpoints, so `FrozenCLIPEmbedder`/`FrozenOpenCLIPEmbedder2` load from
    # disk instead of trying the network.
    ckpt_pth = types.ModuleType("CKPT_PTH")
    ckpt_pth.SDXL_CLIP1_PATH = args.clip_l
    ckpt_pth.SDXL_CLIP2_CKPT_PTH = args.clip_bigg
    ckpt_pth.LLAVA_CLIP_PATH = None
    ckpt_pth.LLAVA_MODEL_PATH = None
    sys.modules["CKPT_PTH"] = ckpt_pth

    _patch_openclip_lazy_text_load()
    _patch_vae_attn_without_xformers()

    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config

    cfg_path = os.path.join(args.supir_src, "options", "SUPIR_v0.yaml")
    config = OmegaConf.load(cfg_path)
    config.model.params.ae_dtype = "fp32"
    config.model.params.diffusion_dtype = "fp32"

    # `FrozenCLIPEmbedder`/`FrozenOpenCLIPEmbedder2` (`sgm/modules/encoders/
    # modules.py`) both default their OWN constructor's `device="cuda"`, and
    # the YAML's `conditioner_config.params.emb_models` entries never override
    # it - so on a CPU-only machine `.to(self.device)` inside `forward()`
    # lazily inits a CUDA context and raises `RuntimeError: Found no NVIDIA
    # driver`. Neither the `ae_dtype`/`diffusion_dtype` overrides above nor
    # `model.cpu()` touch this - `.cpu()` moves already-constructed PARAMETERS,
    # it does not rewrite a stored `self.device` string a later forward reads.
    for emb in config.model.params.conditioner_config.params.emb_models:
        if emb.target in (
            "sgm.modules.encoders.modules.FrozenCLIPEmbedder",
            "sgm.modules.encoders.modules.FrozenOpenCLIPEmbedder2",
        ):
            emb.params.device = "cpu"

    print("instantiating SUPIRModel (CLIP towers load real weights here) ...", flush=True)
    model = instantiate_from_config(config.model).cpu()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"streaming SDXL base weights from {args.sdxl} ...", flush=True)
    missing_sdxl, unexpected_sdxl = load_state_dict_streaming(model, args.sdxl)
    print(f"  {len(missing_sdxl)} missing (expected: the SUPIR-only delta), "
          f"{len(unexpected_sdxl)} unexpected", flush=True)
    if unexpected_sdxl:
        print(f"  unexpected (SDXL): {unexpected_sdxl[:20]}", flush=True)

    print(f"streaming SUPIR-v0Q weights from {args.supir} ...", flush=True)
    missing_supir, unexpected_supir = load_state_dict_streaming(model, args.supir)
    print(f"  {len(missing_supir)} missing (expected: the frozen SDXL backbone, "
          f"already loaded above), {len(unexpected_supir)} unexpected", flush=True)

    # Self-check: name stray tensors, don't silently drop them.
    # `model.control_model.mask_LQ` is the one tensor in the real checkpoint
    # with no counterpart in the released `GLVControl` code (a leftover from
    # an unreleased masking variant) - it MUST show up here, named, not vanish.
    assert "model.control_model.mask_LQ" in unexpected_supir, (
        "expected the documented stray tensor 'model.control_model.mask_LQ' "
        f"in the SUPIR checkpoint's unexpected keys; got: {unexpected_supir}"
    )
    print(f"  confirmed stray tensor present and rejected: model.control_model.mask_LQ "
          f"({len(unexpected_supir)} unexpected total)", flush=True)

    return model, config, {
        "missing_sdxl": len(missing_sdxl), "unexpected_sdxl": len(unexpected_sdxl),
        "missing_supir": len(missing_supir), "unexpected_supir": len(unexpected_supir),
    }


# ---------------------------------------------------------------------------
# Hook plumbing
# ---------------------------------------------------------------------------
def tap_once(store, mod, name, want_input=False, list_output=False):
    """Forward hook that fires exactly once then removes itself - used for
    the trunk/adaptor/final-UNet taps, where only ONE representative forward
    (the first denoiser evaluation, batch = [uncond; cond] stacked by the
    CFG guider's `prepare_inputs`) is wanted, not one copy per sampling step.
    """
    handle_box = {}

    def fn(_mod, inp, outp):
        if list_output:
            for i, o in enumerate(outp):
                record(store, f"{name}{i}", o)
        else:
            record(store, name, outp)
        if want_input:
            for i, t in enumerate(inp):
                record(store, f"{name}.in{i}", t)
        handle_box["h"].remove()

    handle_box["h"] = mod.register_forward_hook(fn)


def tap_accumulate(mod, sink_list):
    """Forward hook that appends every (input, output) pair - used where the
    module is legitimately called more than once with DIFFERENT real inputs
    in one pipeline run (e.g. quant_conv for `_z`'s moments AND for
    `z_stage1`'s moments; each CLIP embedder for the positive AND the
    negative prompt) and every call matters.
    """

    def fn(_mod, inp, outp):
        # Not every input is a tensor - `FrozenCLIPEmbedder.forward(text)`
        # takes a raw `List[str]` (it tokenizes internally), so clone only the
        # tensor positions and pass anything else through unchanged (a Python
        # list of strings is immutable-enough for this dumper's purposes; it
        # is never mutated after the call it was captured from).
        snapshot = tuple(t.detach().clone() if torch.is_tensor(t) else t for t in inp)
        sink_list.append((snapshot, outp))

    return mod.register_forward_hook(fn)


# ---------------------------------------------------------------------------
# The pipeline itself - mirrors `SUPIR_model.py::SUPIRModel.batchify_sample`
# line for line (same real methods, same order, same seeding call), so every
# intermediate is a REAL top-level API return value rather than a hand
# re-derivation. Kept as a free function (not a call into the black-box
# `batchify_sample`) purely so `_z`/`x_stage1`/`z_stage1`/`noised_z`/
# `_samples` - which `batchify_sample` computes but does not return - are
# reachable directly.
# ---------------------------------------------------------------------------
def run_pipeline(model, x, args, store, manifest):
    from pytorch_lightning import seed_everything
    from sgm.util import instantiate_from_config
    from sgm.modules.diffusionmodules.sampling import RestoreEDMSampler
    from sgm.modules.diffusionmodules.guiders import LinearCFG
    from SUPIR.utils.colorfix import wavelet_reconstruction

    # ---- fire-once structural taps (registered before ANY forward pass) ---
    tap_once(store, model.model.control_model, "trunk.hs", list_output=True)
    for i, pm in enumerate(model.model.diffusion_model.project_modules):
        tap_once(store, pm, f"proj{i}", want_input=True)
    tap_once(store, model.model.diffusion_model, "unet_final_raw_out")

    # ---- accumulating taps (fire on EVERY real call; index by call order) -
    quant_conv_calls, post_quant_conv_calls, decoder_calls = [], [], []
    denoise_encoder_calls, encoder_calls = [], []
    h_quant = tap_accumulate(model.first_stage_model.quant_conv, quant_conv_calls)
    h_post_quant = tap_accumulate(model.first_stage_model.post_quant_conv, post_quant_conv_calls)
    h_decoder = tap_accumulate(model.first_stage_model.decoder, decoder_calls)
    h_denoise_enc = tap_accumulate(model.first_stage_model.denoise_encoder, denoise_encoder_calls)
    h_encoder = tap_accumulate(model.first_stage_model.encoder, encoder_calls)

    clip_l_calls, clip_bigg_calls = [], []
    h_clip_l = tap_accumulate(model.conditioner.embedders[0], clip_l_calls)
    h_clip_bigg = tap_accumulate(model.conditioner.embedders[1], clip_bigg_calls)

    # ---- per-step sampler taps: class-level patches, restored afterward ---
    # `sampler_step`/`denoise`/`LinearCFG.__call__` compute the ramped
    # control-scale and CFG-scale as LOCAL variables the sampler never
    # returns; wrapping the class methods (not the not-yet-built instance) is
    # the only way to see them without duplicating the sampler's own math by
    # hand ahead of time - see the two-independent-ways asserts below, which
    # DO duplicate the math, but only to check it against what these taps
    # observed the real run actually use.
    step_taps = []
    denoise_taps = []
    cfg_taps = []
    orig_sampler_step = RestoreEDMSampler.sampler_step
    orig_denoise = RestoreEDMSampler.denoise
    orig_guider_call = LinearCFG.__call__

    def patched_sampler_step(self, sigma, next_sigma, denoiser, x_, cond, uc=None, gamma=0.0,
                              x_center=None, eps_noise=None, control_scale=1.0,
                              use_linear_control_scale=False, control_scale_start=0.0):
        x_in = x_.detach().clone()
        out = orig_sampler_step(self, sigma, next_sigma, denoiser, x_, cond, uc, gamma, x_center,
                                 eps_noise, control_scale, use_linear_control_scale, control_scale_start)
        step_taps.append({
            "sigma": sigma.detach().clone(), "next_sigma": next_sigma.detach().clone(),
            "gamma": float(gamma), "x_in": x_in, "x_out": out.detach().clone(),
            "control_scale_in": float(control_scale), "control_scale_start": float(control_scale_start),
            "use_linear_control_scale": bool(use_linear_control_scale),
        })
        return out

    def patched_denoise(self, x_, denoiser, sigma, cond, uc, control_scale=1.0):
        out = orig_denoise(self, x_, denoiser, sigma, cond, uc, control_scale=control_scale)
        denoise_taps.append({"sigma_hat": sigma.detach().clone(), "control_scale_used": float(control_scale)})
        return out

    def patched_guider_call(self, x_, sigma):
        scale_value = self.scale_schedule(sigma)
        cfg_taps.append({
            "sigma": sigma.detach().clone(),
            "cfg_scale_used": scale_value.detach().clone() if isinstance(scale_value, torch.Tensor)
            else torch.tensor([float(scale_value)]),
        })
        return orig_guider_call(self, x_, sigma)

    RestoreEDMSampler.sampler_step = patched_sampler_step
    RestoreEDMSampler.denoise = patched_denoise
    LinearCFG.__call__ = patched_guider_call

    try:
        # ---- sampler_config overrides, exactly as batchify_sample does ----
        model.sampler_config.params.num_steps = args.steps
        model.sampler_config.params.guider_config.params.scale_min = CFG_SCALE
        model.sampler_config.params.guider_config.params.scale = CFG_SCALE_START
        model.sampler_config.params.restore_cfg = RESTORATION_SCALE
        model.sampler_config.params.s_churn = args.s_churn
        model.sampler_config.params.s_noise = S_NOISE
        # `BaseDiffusionSampler.__init__` (sampling.py) also defaults its OWN
        # `device="cuda"` with no YAML override, same class of bug as the two
        # CLIP embedders above - `LegacyDDPMDiscretization.get_sigmas` builds
        # its schedule tensors on `self.device`, so this crashes the same way
        # on a CPU-only machine if left unset.
        model.sampler_config.params.device = "cpu"
        model.sampler = instantiate_from_config(model.sampler_config)

        seed_everything(SEED)

        record(store, "in.lq_image", x)

        print("encode_first_stage_with_denoise (the HINT, _z) ...", flush=True)
        _z = model.encode_first_stage_with_denoise(x, use_sample=False)
        record(store, "stage1._z", _z)

        print("decode_first_stage(_z) -> x_stage1 (frozen decoder) ...", flush=True)
        x_stage1 = model.decode_first_stage(_z)
        record(store, "stage1.x_stage1", x_stage1)

        print("encode_first_stage(x_stage1) -> z_stage1 (clean re-encode, REAL .sample()) ...", flush=True)
        z_stage1 = model.encode_first_stage(x_stage1)
        record(store, "stage1.z_stage1", z_stage1)

        print("prepare_condition (both CLIP towers, real tokenizer + real weights) ...", flush=True)
        c, uc = model.prepare_condition(_z, [PROMPT], model.p_p, model.n_p, 1)
        record(store, "cond.c.crossattn", c["crossattn"])
        record(store, "cond.c.vector", c["vector"])
        record(store, "cond.uc.crossattn", uc["crossattn"])
        record(store, "cond.uc.vector", uc["vector"])
        record(store, "cond.c.control", c["control"])
        record(store, "cond.uc.control", uc["control"])
        # Self-check: the roadmap states the unconditional branch keeps the
        # SAME LQ control latent as the conditional branch - only the text
        # differs. Assert it rather than take the docstring's word for it.
        assert torch.equal(c["control"], uc["control"]), (
            "expected c['control'] == uc['control'] (same LQ hint, only text differs)"
        )

        denoiser = lambda inp, sigma, cnd, control_scale: model.denoiser(
            model.model, inp, sigma, cnd, control_scale
        )

        noised_z = torch.randn_like(_z)
        record(store, "in.noised_z", noised_z)

        print(f"sampling ({args.steps} steps, RestoreEDMSampler) ...", flush=True)
        _samples = model.sampler(
            denoiser, noised_z, cond=c, uc=uc, x_center=z_stage1,
            control_scale=CONTROL_SCALE, use_linear_control_scale=True,
            control_scale_start=CONTROL_SCALE_START,
        )
        record(store, "out.samples_latent", _samples)

        print("decode_first_stage(_samples) -> final decode ...", flush=True)
        samples = model.decode_first_stage(_samples)
        record(store, "out.decoded", samples)

        print("wavelet_reconstruction(samples, x_stage1) -> colour-fixed out ...", flush=True)
        out_colorfixed = wavelet_reconstruction(samples, x_stage1)
        record(store, "out.colorfixed", out_colorfixed)
    finally:
        RestoreEDMSampler.sampler_step = orig_sampler_step
        RestoreEDMSampler.denoise = orig_denoise
        LinearCFG.__call__ = orig_guider_call
        for h in (h_quant, h_post_quant, h_decoder, h_denoise_enc, h_encoder, h_clip_l, h_clip_bigg):
            h.remove()

    # ---- unpack the accumulating taps by call order -----------------------
    # quant_conv fires for: [0] _z's moments, [1] z_stage1's moments.
    assert len(quant_conv_calls) == 2, f"expected 2 quant_conv calls, got {len(quant_conv_calls)}"
    record(store, "stage1._z.moments", quant_conv_calls[0][1])
    record(store, "stage1.z_stage1.moments", quant_conv_calls[1][1])
    # post_quant_conv/decoder fire for: [0] x_stage1's decode, [1] the final decode.
    assert len(decoder_calls) == 2, f"expected 2 decoder calls, got {len(decoder_calls)}"
    record(store, "stage1.x_stage1.pre_decoder", post_quant_conv_calls[0][1])
    record(store, "out.decoded.pre_decoder", post_quant_conv_calls[1][1])
    # denoise_encoder fires once (only used for _z); encoder fires once (only
    # used for z_stage1 - a SEPARATE frozen module, byte-identical topology
    # to denoise_encoder, only the weights differ).
    assert len(denoise_encoder_calls) == 1 and len(encoder_calls) == 1
    record(store, "stage1._z.denoise_encoder_out", denoise_encoder_calls[0][1])
    record(store, "stage1.z_stage1.encoder_out", encoder_calls[0][1])

    # `z_stage1.mode` computed independently from the SAME tapped moments, for
    # side-by-side comparison against the real (sampled) `stage1.z_stage1` -
    # see the module docstring's note on the roadmap discrepancy.
    moments = quant_conv_calls[1][1]
    mean, _logvar = torch.chunk(moments, 2, dim=1)
    record(store, "stage1.z_stage1.mode", model.scale_factor * mean)

    # CLIP towers: [0] = positive (caption + p_p), [1] = negative (n_p alone).
    assert len(clip_l_calls) == 2 and len(clip_bigg_calls) == 2
    record(store, "cond.clip_l.pos", clip_l_calls[0][1])
    record(store, "cond.clip_l.neg", clip_l_calls[1][1])
    bigg_pos_crossattn, bigg_pos_pooled = clip_bigg_calls[0][1]
    bigg_neg_crossattn, bigg_neg_pooled = clip_bigg_calls[1][1]
    record(store, "cond.clip_bigg.pos.crossattn", bigg_pos_crossattn)
    record(store, "cond.clip_bigg.pos.pooled", bigg_pos_pooled)
    record(store, "cond.clip_bigg.neg.crossattn", bigg_neg_crossattn)
    record(store, "cond.clip_bigg.neg.pooled", bigg_neg_pooled)

    # ---- per-step trajectory: sigma / next_sigma / gamma / control-scale /
    #      cfg-scale / pre+post latent, one row per sampler step -----------
    assert len(step_taps) == len(denoise_taps) == len(cfg_taps) == args.steps, (
        f"expected {args.steps} rows in each per-step tap, got "
        f"{len(step_taps)}/{len(denoise_taps)}/{len(cfg_taps)}"
    )
    sigmas_rows, next_sigmas_rows, gammas = [], [], []
    control_scale_used_rows, cfg_scale_used_rows = [], []
    for i, (st, dt, ct) in enumerate(zip(step_taps, denoise_taps, cfg_taps)):
        record(store, f"step{i}.x_in", st["x_in"])
        record(store, f"step{i}.x_out", st["x_out"])
        sigmas_rows.append(st["sigma"])
        next_sigmas_rows.append(st["next_sigma"])
        gammas.append(st["gamma"])
        control_scale_used_rows.append(dt["control_scale_used"])
        cfg_scale_used_rows.append(float(ct["cfg_scale_used"].reshape(-1)[0]))

        # Self-validation #1 (playbook 1: two independent ways, asserted
        # before anything is written to disk): the control-scale ramp
        # `sampler_step` applies internally, recomputed by hand from the same
        # per-step sigma this tap observed, must match what `denoise` (the
        # very next call inside that same `sampler_step`) actually received.
        sigma0 = float(st["sigma"].reshape(-1)[0])
        if st["use_linear_control_scale"]:
            hand_ramped = ((sigma0 / model.sampler.sigma_max)
                           * (st["control_scale_start"] - st["control_scale_in"])
                           + st["control_scale_in"])
        else:
            hand_ramped = st["control_scale_in"]
        assert abs(hand_ramped - dt["control_scale_used"]) < 1e-6, (
            f"step {i}: hand-ramped control_scale {hand_ramped} != observed "
            f"{dt['control_scale_used']}"
        )

        # Self-validation #2: same idea for the CFG guider's own linear ramp
        # (`LinearCFG.scale_schedule`), hand-recomputed from the CFG config
        # this run set (`CFG_SCALE`/`CFG_SCALE_START`) against the value the
        # real `LinearCFG.__call__` observed.
        #
        # `LinearCFG.__init__(self, scale, scale_min=None, ...)` names its
        # ctor args the OPPOSITE way the module docstring's prose does: the
        # ctor's `scale` is the ramp's value AT sigma_max (so it gets
        # CFG_SCALE_START here), and `scale_min` is the value AT sigma -> 0
        # (so it gets CFG_SCALE, the higher Quality-preset target) - see the
        # `guider_config.params` overrides above. Get this backwards here and
        # the check still "passes" numerically wrong ranges undetected, so
        # the ctor mapping is spelled out rather than trusted from memory.
        #
        # Also: `ct["sigma"]` is `sigma_hat` (the CHURNED sigma `denoise`
        # actually calls the guider with), not the raw schedule `sigma` -
        # with `s_churn > 0` these differ, and at step 0 with a short
        # schedule `sigma_hat` can exceed `sigma_max` (gamma capped at
        # sqrt(2)-1), which is expected, not a bug.
        cfg_sigma0 = float(ct["sigma"].reshape(-1)[0])
        hand_cfg = (CFG_SCALE_START - CFG_SCALE) * cfg_sigma0 / model.sampler.sigma_max + CFG_SCALE
        assert abs(hand_cfg - cfg_scale_used_rows[-1]) < 1e-5, (
            f"step {i}: hand-ramped cfg scale {hand_cfg} != observed {cfg_scale_used_rows[-1]}"
        )

    record(store, "steps.sigma", torch.cat([s.reshape(-1) for s in sigmas_rows]))
    record(store, "steps.next_sigma", torch.cat([s.reshape(-1) for s in next_sigmas_rows]))
    record(store, "steps.gamma", torch.tensor(gammas, dtype=torch.float32))
    record(store, "steps.control_scale_used", torch.tensor(control_scale_used_rows, dtype=torch.float32))
    record(store, "steps.cfg_scale_used", torch.tensor(cfg_scale_used_rows, dtype=torch.float32))

    # ---- sigma schedule: sampler's own path vs a hand-computed path -------
    # Self-validation #3: `sqrt((1-abar)/abar)` from a hand-rolled linear-beta
    # schedule (independent of `sgm.modules.diffusionmodules.util.
    # make_beta_schedule`, which the real discretizer calls), compared to
    # what `sampler.discretization` - the exact call `prepare_sampling_loop`
    # makes - actually produced.
    sigmas_real = model.sampler.discretization(model.sampler.num_steps, device="cpu")
    sigmas_hand = _hand_sigma_schedule(model.sampler.num_steps)
    assert torch.allclose(sigmas_real, sigmas_hand, atol=1e-3, rtol=1e-3), (
        f"sigma schedule mismatch: sampler={sigmas_real} hand={sigmas_hand}"
    )
    record(store, "schedule.sigmas_sampler", sigmas_real)
    record(store, "schedule.sigmas_hand", sigmas_hand)
    record(store, "schedule.sigma_max", torch.tensor([model.sampler.sigma_max], dtype=torch.float32))

    manifest["params"]["self_checks_passed"] = [
        "control_scale_ramp", "cfg_scale_ramp", "sigma_schedule_two_ways",
        "c_uc_control_identical", "mask_LQ_present_in_unexpected_keys",
    ]


def _hand_sigma_schedule(num_steps, linear_start=0.00085, linear_end=0.0120, num_timesteps=1000):
    """Independent re-derivation of `LegacyDDPMDiscretization`'s sigma
    schedule: a plain linear-beta DDPM schedule, `sigma = sqrt((1-abar)/abar)`,
    subsampled to `num_steps` roughly-equally-spaced timesteps, flipped
    descending (sigma_max first), with a trailing 0.0 appended - mirrors
    `sgm/modules/diffusionmodules/discretizer.py` structurally but is written
    from the formula, not by calling that module, so it is a genuine second
    path, not a re-import of the first.
    """
    betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=np.float64) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    if num_steps < num_timesteps:
        idx = np.linspace(num_timesteps - 1, 0, num_steps, endpoint=False).astype(int)[::-1]
        alphas_cumprod = alphas_cumprod[idx]
    sigmas = np.sqrt((1.0 - alphas_cumprod) / alphas_cumprod)
    sigmas = sigmas[::-1].copy()  # descending: sigma_max first
    sigmas = np.concatenate([sigmas, [0.0]])
    return torch.tensor(sigmas, dtype=torch.float32)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--supir-src", required=True, help="local clone of Fanghua-Yu/SUPIR (see module docstring for the pinned commit)")
    ap.add_argument("--sdxl", required=True, help="sd_xl_base_1.0_0.9vae.safetensors")
    ap.add_argument("--supir", required=True, help="SUPIR-v0Q_fp32.safetensors")
    ap.add_argument("--clip-l", required=True, help="openai/clip-vit-large-patch14 dir (config.json + model.safetensors + tokenizer files)")
    ap.add_argument("--clip-bigg", required=True, help="open_clip_model.safetensors (laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent", type=int, default=LATENT_DEFAULT, help="latent H=W (32 => a 256x256 LQ image)")
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT, help="sampler steps (small: this gates composition, not native step count)")
    ap.add_argument("--s-churn", type=float, default=S_CHURN_DEFAULT, dest="s_churn",
                     help="RestoreEDMSampler's s_churn (default 5.0, matching the committed testdata/supir/ "
                          "golden exactly - a re-run with no override reproduces it bit-for-bit). Pass 0.0 to "
                          "make gamma identically zero (sigma_hat == sigma), so the forward is deterministic "
                          "from the seed alone with no unrecoverable churn-noise draw - the shape a Rust-side "
                          "forward replay needs to compare against exactly. See the module docstring's "
                          "'first task' note for why the DEFAULT golden cannot be replayed.")
    ap.add_argument("--name", default="stages.safetensors")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    model, config, load_stats = build_model(args)

    store = {}
    mpath = os.path.join(args.out, "manifest.json")
    manifest = {"files": {}, "params": {}}

    g = torch.Generator().manual_seed(SEED)
    px = 8 * args.latent  # VAE downsamples by exactly 8
    x = torch.rand(1, 3, px, px, generator=g) * 2.0 - 1.0  # [-1, 1], RGB, like a real preprocessed LQ image

    run_pipeline(model, x, args, store, manifest)

    save(store, args.out, args.name, manifest)

    from golden_source import source_block  # noqa: E402  (tools/goldens dir is on sys.path[0])

    manifest["source"] = source_block(
        checkpoint="stabilityai/stable-diffusion-xl-base-1.0 + yushan777/SUPIR (SUPIR-v0Q_fp32) "
                   "+ openai/clip-vit-large-patch14 + laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        files=[args.sdxl, args.supir],
        hash_files=False,  # ~12 GB combined; hashing would dominate this dumper's runtime for no test benefit
        identity={
            "model_channels": 320,
            "context_dim": 2048,
            "adm_in_channels": 2816,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "channel_mult_0": 1, "channel_mult_1": 2, "channel_mult_2": 4,
            "transformer_depth_0": 1, "transformer_depth_1": 2, "transformer_depth_2": 10,
            "project_channel_scale": 2,
            "n_project_modules": 12,
            "n_trunk_outputs": 10,
            "latent": args.latent,
            "steps": args.steps,
        },
    )
    manifest["params"].update({
        "seed": SEED,
        "prompt": PROMPT,
        "torch": torch.__version__,
        "cfg_scale": CFG_SCALE, "cfg_scale_start": CFG_SCALE_START,
        "control_scale": CONTROL_SCALE, "control_scale_start": CONTROL_SCALE_START,
        "restoration_scale": RESTORATION_SCALE, "s_churn": args.s_churn, "s_noise": S_NOISE,
        "supir_upstream_commit": "bda91af2000042f8bedfec8897d92917e67c1d88",
        "load_stats": load_stats,
    })
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
