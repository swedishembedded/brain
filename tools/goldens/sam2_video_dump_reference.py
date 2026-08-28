#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump SAM 2 VIDEO-path reference goldens: the temporal memory bank.

The image path is already covered by `sam2_dump_reference.py`. This script
covers what that one declares out of scope - `memory_attention`,
`memory_encoder`, the object-pointer temporal encoding and the propagation
loop that conditions frame t on the memory of previous frames - by running the
OFFICIAL `facebookresearch/sam2` package (loaded with `strict=True`, so a
rename this script gets wrong is a hard error, never a random weight) against
the released `sam2.1_hiera_*.pt` checkpoint.

Three fixture files, three levels of the parity ladder:

  memattn.safetensors  `MemoryAttention` on SEEDED RANDOM inputs: the current
                       frame's [HW, C] features and their sine position
                       encoding, a memory of `num_maskmem` spatial slabs plus
                       object-pointer tokens, and every per-layer output. Random
                       inputs are deliberate - they are maximally sensitive to a
                       transposed RoPE pair or an off-by-one in `rope_k_repeat`.
  memenc.safetensors   `MemoryEncoder` on seeded random `pix_feat` and mask
                       logits, with the mask-downsampler, fuser and out_proj
                       taps split out.
  video.safetensors    an END-TO-END propagation: `SAM2VideoPredictor` over a
                       short synthetic clip, one point prompt on one frame,
                       every frame's memory-conditioned feature, mask logits,
                       object pointer, object score and encoded memory.

  manifest.json        shapes, sha256 per file, the reference config, the
                       weight-health report (see below) and library versions.

WEIGHT HEALTH. A published checkpoint whose tensors are all at PyTorch default
init - dropped at load under `strict=False` because of a key-name mismatch - is
a real failure mode and it is silent. So the manifest records, per video-path
tensor, `max|w|` against `1/sqrt(fan_in)` and the excess kurtosis: uniform
default init sits at kurtosis ~1.8 with `max|w| == 1/sqrt(fan_in)`. The Rust
side asserts the same thing on the loaded weights, so the check runs in CI and
not only here.

Usage:
  python tools/goldens/sam2_video_dump_reference.py \
      --code   <sam2 repo root, the dir containing the `sam2` package> \
      --config <sam2/configs/sam2.1/sam2.1_hiera_t.yaml> \
      --ckpt   <sam2.1_hiera_tiny.pt> \
      --out    testdata/sam2/hiera-tiny [--frames 6] [--seed 42]
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUBBED = False


def _image_dumper():
    """Reuse the image dumper's hydra-free `instantiate` / dep stubs."""
    spec = importlib.util.spec_from_file_location(
        "sam2_dump_reference", os.path.join(_HERE, "sam2_dump_reference.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(code, config, ckpt, predictor):
    """The official package, instantiated from its own YAML, weights strict."""
    import yaml

    global _STUBBED
    sys.path.insert(0, os.path.abspath(code))
    base = _image_dumper()
    if not _STUBBED:
        # Stubbing is a global, one-shot mutation of `sys.modules`; running it
        # twice makes `find_spec` choke on the stub it installed itself.
        base.stub_optional_deps()
        _STUBBED = True
    with open(config) as f:
        cfg = yaml.safe_load(f)["model"]
    if predictor:
        cfg = dict(cfg, _target_="sam2.sam2_video_predictor.SAM2VideoPredictor")
    model = base.instantiate(cfg).float().eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(sd, strict=True)
    return model, cfg, sd


# --------------------------------------------------------------------------- #
# weight health
# --------------------------------------------------------------------------- #
VIDEO_ONLY = ("memory_attention.", "memory_encoder.", "maskmem_tpos_enc",
              "no_mem_embed", "no_mem_pos_enc", "mask_downsample.",
              "obj_ptr_tpos_proj.", "no_obj_embed_spatial")


def weight_health(sd):
    """Per video-path tensor: is it distinguishable from PyTorch default init?

    `nn.Linear`/`nn.Conv2d` default to U(-k, k) with k = 1/sqrt(fan_in), whose
    excess kurtosis is -1.2 (kurtosis 1.8 unadjusted) and whose `max|w|` sits
    essentially AT k. A trained tensor is heavier-tailed and its max is a
    fraction of k. Both numbers are recorded so the Rust gate can assert the
    same property without re-deriving the statistic.
    """
    out = {}
    for name, w in sd.items():
        if not name.startswith(VIDEO_ONLY):
            continue
        x = w.detach().float().flatten()
        if x.numel() < 8:
            continue
        fan_in = int(w.shape[1]) if w.dim() >= 2 else 0
        if w.dim() == 4:  # conv: fan_in = Cin/groups * kh * kw
            fan_in = int(w.shape[1] * w.shape[2] * w.shape[3])
        k = (1.0 / fan_in ** 0.5) if fan_in else float("nan")
        m = x.mean()
        s = x.std(unbiased=False)
        kurt = float((((x - m) / s) ** 4).mean()) if float(s) > 0 else float("nan")
        out[name] = {
            "numel": int(x.numel()),
            "fan_in": fan_in,
            "std": float(s),
            "max_abs": float(x.abs().max()),
            "default_init_bound": k,
            "kurtosis": kurt,
        }
    return out


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(out, name, tensors, index):
    path = os.path.join(out, name)
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, path)
    index[name] = {
        "sha256": sha256(path),
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
    }
    print(f"  wrote {name}: {len(tensors)} tensors", flush=True)


def dump_memory_attention(model, out, index, gen):
    """`MemoryAttention` on seeded random inputs, with per-layer taps."""
    C = model.hidden_dim
    mem_dim = model.mem_dim
    hw = model.image_size // model.backbone_stride
    n_cur = hw * hw
    n_mask = model.num_maskmem            # spatial memory slabs
    n_ptr_tokens = 4 * (C // mem_dim)     # 4 pointers, each split into C/mem_dim tokens
    n_spatial = n_mask * n_cur

    curr = torch.randn(n_cur, 1, C, generator=gen) * 0.5
    curr_pos = torch.randn(n_cur, 1, C, generator=gen) * 0.5
    memory = torch.randn(n_spatial + n_ptr_tokens, 1, mem_dim, generator=gen) * 0.5
    memory_pos = torch.randn(n_spatial + n_ptr_tokens, 1, mem_dim, generator=gen) * 0.5

    taps = {}
    handles = []
    for i, layer in enumerate(model.memory_attention.layers):
        handles.append(layer.register_forward_hook(
            lambda m, inp, o, i=i: taps.__setitem__(f"memattn_layer{i}_out", o.detach().clone())))
    with torch.inference_mode():
        y = model.memory_attention(
            curr=[curr], memory=memory, curr_pos=[curr_pos],
            memory_pos=memory_pos, num_obj_ptr_tokens=n_ptr_tokens)
    for h in handles:
        h.remove()

    t = {
        "curr": curr.squeeze(1),
        "curr_pos": curr_pos.squeeze(1),
        "memory": memory.squeeze(1),
        "memory_pos": memory_pos.squeeze(1),
        "num_obj_ptr_tokens": torch.tensor([float(n_ptr_tokens)]),
        "num_maskmem_slabs": torch.tensor([float(n_mask)]),
        "memattn_out": y.squeeze(1),
    }
    # the hooks see [seq, B, C] because MemoryAttention transposes back only at
    # the end; squeeze the batch axis the same way for every tap.
    for k, v in taps.items():
        t[k] = v.squeeze(1) if v.dim() == 3 else v
    write(out, "memattn.safetensors", t, index)


def dump_memory_encoder(model, out, index, gen):
    """`MemoryEncoder` on seeded random inputs, with the three internal taps."""
    C = model.hidden_dim
    hw = model.image_size // model.backbone_stride
    pix_feat = torch.randn(1, C, hw, hw, generator=gen) * 0.5
    mask_logits = torch.randn(1, 1, model.image_size, model.image_size, generator=gen) * 3.0
    # exactly what `_encode_new_memory` feeds it (sigmoid, then scale and bias)
    mask_for_mem = torch.sigmoid(mask_logits) * model.sigmoid_scale_for_mem_enc \
        + model.sigmoid_bias_for_mem_enc

    enc = model.memory_encoder
    taps = {}
    hs = [
        enc.mask_downsampler.register_forward_hook(
            lambda m, i, o: taps.__setitem__("mask_downsampled", o.detach().clone())),
        enc.pix_feat_proj.register_forward_hook(
            lambda m, i, o: taps.__setitem__("pix_feat_proj", o.detach().clone())),
        enc.fuser.register_forward_hook(
            lambda m, i, o: taps.__setitem__("fuser_out", o.detach().clone())),
    ]
    for i, layer in enumerate(enc.fuser.layers):
        hs.append(layer.register_forward_hook(
            lambda m, inp, o, i=i: taps.__setitem__(f"fuser_layer{i}_out", o.detach().clone())))
    with torch.inference_mode():
        res = enc(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
    for h in hs:
        h.remove()

    t = {
        "pix_feat": pix_feat,
        "mask_logits": mask_logits,
        "mask_for_mem": mask_for_mem,
        "sigmoid_scale": torch.tensor([float(model.sigmoid_scale_for_mem_enc)]),
        "sigmoid_bias": torch.tensor([float(model.sigmoid_bias_for_mem_enc)]),
        "maskmem_features": res["vision_features"],
        "maskmem_pos_enc": res["vision_pos_enc"][0],
        **taps,
    }
    write(out, "memenc.safetensors", t, index)


def dump_obj_ptr_tpos(model, out_dict, gen):
    """`get_1d_sine_pe(t/t_diff_max, C) -> obj_ptr_tpos_proj` for every t_diff.

    Small, so it rides along in `video.safetensors`: it is the one piece of the
    pointer path that is pure host arithmetic in the Rust port.
    """
    from sam2.modeling.sam2_utils import get_1d_sine_pe

    t_max = model.max_obj_ptrs_in_encoder - 1
    pos = torch.arange(-t_max, t_max + 1, dtype=torch.float32)
    with torch.inference_mode():
        pe = get_1d_sine_pe(pos / t_max, dim=model.hidden_dim)
        proj = model.obj_ptr_tpos_proj(pe)
    out_dict["objptr_tpos_input"] = pos
    out_dict["objptr_tpos_sine"] = pe
    out_dict["objptr_tpos_proj"] = proj
    out_dict["objptr_t_diff_max"] = torch.tensor([float(t_max)])


def synth_frames(n, h, w, seed, occlude=()):
    """A moving disc on a textured background, briefly hidden behind a bar.

    Two properties the fixture needs, both deliberate:

    * the subject MOVES - a static clip would pass even a port that ignored the
      previous frames entirely and re-segmented from scratch;
    * the subject is fully OCCLUDED on the frames in `occlude` - which is the
      only way to reach `no_obj_embed_spatial` and the `object_score <= 0`
      branch of the tracker. Without it that code is never executed by the
      fixture, so no parity gate can see a change to it (verified: a mutation
      deleting the branch survived a clip with no occlusion).
    """
    rng = np.random.default_rng(seed)
    bg = rng.random((h, w, 3)).astype(np.float32) * 0.35 + 0.1
    yy, xx = np.mgrid[0:h, 0:w]
    frames = []
    for i in range(n):
        cx = w * (0.30 + 0.06 * i)
        cy = h * (0.45 + 0.03 * i)
        r = min(h, w) * 0.16
        disc = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        f = bg.copy()
        f[disc] = np.array([0.92, 0.80, 0.25], dtype=np.float32)
        if i in occlude:
            # An opaque slab over the whole band the disc travels through.
            f[int(h * 0.25):int(h * 0.85), :] = np.array([0.05, 0.06, 0.09], dtype=np.float32)
        frames.append(np.clip(f, 0.0, 1.0))
    return frames, (w * 0.30 + 0.0, h * 0.45)


def dump_video(code, config, ckpt, out, index, n_frames, seed, prompt_frame, occlude):
    """End-to-end propagation through `SAM2VideoPredictor`."""
    from PIL import Image

    model, cfg, _ = build(code, config, ckpt, predictor=True)
    h = w = model.image_size
    frames, (cx, cy) = synth_frames(n_frames, h, w, seed, occlude)

    tmp = os.path.join(out, "_frames")
    os.makedirs(tmp, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray((f * 255.0 + 0.5).astype(np.uint8)).save(
            os.path.join(tmp, f"{i:05d}.jpg"), quality=100, subsampling=0)

    # The predictor STORES `maskmem_features` cast to bfloat16 (see
    # `_run_single_frame_inference` / `_run_memory_encoder`) - a state-size
    # optimisation, not part of `SAM2Base`'s math. Comparing an fp32 port
    # against that stored value would charge the port for the reference's own
    # quantisation, so the memory encoder's true fp32 output (and its two
    # inputs, for bisecting) are captured here as well.
    memenc_calls = []

    def _memenc_pre(_m, args, kwargs):
        pix = args[0] if args else kwargs["pix_feat"]
        msk = args[1] if len(args) > 1 else kwargs["masks"]
        memenc_calls.append({"pix_feat": pix.detach().clone(), "mask_for_mem": msk.detach().clone()})

    def _memenc_post(_m, _a, out):
        memenc_calls[-1]["pre_no_obj"] = out["vision_features"].detach().float().clone()

    h_pre = model.memory_encoder.register_forward_pre_hook(_memenc_pre, with_kwargs=True)
    h_post = model.memory_encoder.register_forward_hook(_memenc_post)

    # The memory encoder's output is NOT the memory. `_encode_new_memory` adds
    # `no_obj_embed_spatial` to it on any frame the object-score head calls
    # occluded, and a module hook cannot see that - it fires one level too deep.
    # Wrapping the method is the level that returns what actually gets stored.
    _orig_encode_new_memory = model._encode_new_memory

    def _wrapped_encode_new_memory(*args, **kwargs):
        feats, pos = _orig_encode_new_memory(*args, **kwargs)
        memenc_calls[-1]["fp32"] = feats.detach().float().clone()
        return feats, pos

    model._encode_new_memory = _wrapped_encode_new_memory

    state = model.init_state(tmp)
    # The JPEG round-trip is invisible to the Rust side: it consumes the
    # NORMALIZED tensors the predictor actually ran on, dumped below.
    images = state["images"].float().clone()

    px, py = float(cx), float(cy)
    with torch.inference_mode():
        model.add_new_points_or_box(
            inference_state=state, frame_idx=prompt_frame, obj_id=1,
            points=np.array([[px, py]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            normalize_coords=True)

    t = {
        "images": images,
        "point_xy": torch.tensor([px, py]),
        "point_label": torch.tensor([1.0]),
        "prompt_frame": torch.tensor([float(prompt_frame)]),
        "num_frames": torch.tensor([float(n_frames)]),
        "video_hw": torch.tensor([float(h), float(w)]),
    }

    seen = {}
    with torch.inference_mode():
        for fidx, _ids, masks in model.propagate_in_video(state):
            seen[int(fidx)] = masks.float().clone()

    h_pre.remove()
    h_post.remove()
    model._encode_new_memory = _orig_encode_new_memory
    # Call order is frame order: preflight encodes the conditioning frame, then
    # `propagate_in_video` walks the remaining frames in ascending order.
    if len(memenc_calls) != n_frames:
        raise SystemExit(f"memory encoder ran {len(memenc_calls)} times for {n_frames} frames")

    obj = state["output_dict_per_obj"][0]
    for fidx in sorted(seen):
        out_f = obj["cond_frame_outputs"].get(fidx) or obj["non_cond_frame_outputs"][fidx]
        t[f"f{fidx}_video_res_masks"] = seen[fidx]
        t[f"f{fidx}_low_res_masks"] = out_f["pred_masks"].float()
        t[f"f{fidx}_obj_ptr"] = out_f["obj_ptr"].float()
        t[f"f{fidx}_object_score_logits"] = out_f["object_score_logits"].float()
        # The bf16-quantised value the reference actually keeps and re-attends
        # to, AND the fp32 the memory encoder produced - the gate uses the
        # second, and the first proves the difference is only the cast.
        t[f"f{fidx}_maskmem_features"] = out_f["maskmem_features"].float()
        c = memenc_calls[fidx]
        t[f"f{fidx}_maskmem_features_fp32"] = c["fp32"]
        # The same value BEFORE `no_obj_embed_spatial` - identical to the one
        # above on any frame the object is visible on, and the witness that the
        # occlusion branch ran on the frames it is not.
        t[f"f{fidx}_maskmem_features_pre_no_obj"] = c["pre_no_obj"]
        t[f"f{fidx}_memenc_pix_feat"] = c["pix_feat"].float()
        t[f"f{fidx}_memenc_mask_for_mem"] = c["mask_for_mem"].float()
        t[f"f{fidx}_maskmem_pos_enc"] = out_f["maskmem_pos_enc"][0].float()
        t[f"f{fidx}_is_cond"] = torch.tensor([1.0 if fidx in obj["cond_frame_outputs"] else 0.0])

    dump_obj_ptr_tpos(model, t, torch.Generator().manual_seed(seed))
    write(out, "video.safetensors", t, index)

    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="sam2 repo root (dir containing the `sam2` package)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--prompt-frame", type=int, default=0)
    ap.add_argument("--occlude", default="", help="comma-separated frames on which the subject is hidden")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    index = {}

    print("== submodule fixtures", flush=True)
    model, cfg, sd = build(a.code, a.config, a.ckpt, predictor=False)
    gen = torch.Generator().manual_seed(a.seed)
    dump_memory_attention(model, a.out, index, gen)
    dump_memory_encoder(model, a.out, index, gen)
    health = weight_health(sd)
    del model, sd

    print("== end-to-end propagation", flush=True)
    occlude = tuple(int(x) for x in a.occlude.split(",") if x.strip())
    vmodel, _ = dump_video(a.code, a.config, a.ckpt, a.out, index,
                           a.frames, a.seed, a.prompt_frame, occlude)

    from golden_source import source_block  # noqa: E402  (tools/goldens dir is on sys.path[0])

    manifest = {
        "files": index,
        "weight_health": health,
        "source": source_block(
            checkpoint="facebook/sam2.1",
            files=[a.ckpt],
            identity={
                "image_size": int(vmodel.image_size),
                "backbone_stride": int(vmodel.backbone_stride),
                "hidden_dim": int(vmodel.hidden_dim),
                "mem_dim": int(vmodel.mem_dim),
                "num_maskmem": int(vmodel.num_maskmem),
                "max_obj_ptrs_in_encoder": int(vmodel.max_obj_ptrs_in_encoder),
                "memory_attention_layers": len(vmodel.memory_attention.layers),
                "memory_attention_ff": int(vmodel.memory_attention.layers[0].dim_feedforward),
            },
        ),
        "params": {
            "image_size": vmodel.image_size,
            "backbone_stride": vmodel.backbone_stride,
            "hidden_dim": vmodel.hidden_dim,
            "mem_dim": vmodel.mem_dim,
            "num_maskmem": vmodel.num_maskmem,
            "max_obj_ptrs_in_encoder": vmodel.max_obj_ptrs_in_encoder,
            "memory_temporal_stride_for_eval": vmodel.memory_temporal_stride_for_eval,
            "sigmoid_scale_for_mem_enc": vmodel.sigmoid_scale_for_mem_enc,
            "sigmoid_bias_for_mem_enc": vmodel.sigmoid_bias_for_mem_enc,
            "memory_attention_layers": len(vmodel.memory_attention.layers),
            "memory_attention_ff": vmodel.memory_attention.layers[0].dim_feedforward,
            "add_tpos_enc_to_obj_ptrs": vmodel.add_tpos_enc_to_obj_ptrs,
            "proj_tpos_enc_in_obj_ptrs": vmodel.proj_tpos_enc_in_obj_ptrs,
            "use_signed_tpos_enc_to_obj_ptrs": vmodel.use_signed_tpos_enc_to_obj_ptrs,
            "only_obj_ptrs_in_the_past_for_eval": vmodel.only_obj_ptrs_in_the_past_for_eval,
            "no_obj_embed_spatial": vmodel.no_obj_embed_spatial is not None,
            "maskmem_features_stored_as": "bfloat16",
            "multimask_output_for_tracking": vmodel.multimask_output_for_tracking,
            "frames": a.frames,
            "prompt_frame": a.prompt_frame,
            "occluded_frames": list(occlude),
            "seed": a.seed,
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    with open(os.path.join(a.out, "video_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f"== wrote {a.out}/video_manifest.json", flush=True)


if __name__ == "__main__":
    main()
