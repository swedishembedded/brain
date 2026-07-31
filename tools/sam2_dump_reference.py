#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump SAM 2 (image path) reference goldens for brain's parity ladder.

Runs the OFFICIAL facebookresearch/sam2 model (CPU, fp32, fixed seed, fixed
synthetic image + fixed prompts) and saves, per stage, everything a Rust port
needs to bisect a forward-parity failure:

  input.safetensors      raw RGB test image + the normalized model input
  trunk.safetensors      Hiera: interpolated pos_embed, patch_embed output,
                         per-tapped-block output (stage ends, q_pool blocks,
                         global-attention blocks), the 4 trunk feature maps
  neck.safetensors       FPN: per-level lateral conv output, top-down fused
                         output, PositionEmbeddingSine per level, post-scalp
                         backbone_fpn, conv_s0/conv_s1 high-res features and
                         the final image embedding (+ no_mem_embed)
  case_<name>.safetensors prompt encoder I/O (sparse/dense/dense_pe) and the
                         two-way mask decoder taps (tokens, per-layer
                         queries/keys, hs, src, upscaled embedding, hyper_in,
                         mask logits, IoU, object score, hi-res masks) for one
                         prompt case
  manifest.json          every tensor's shape + dtype, sha256 per file, the
                         exact reference config, the per-block trunk table,
                         the self-check diffs and library versions
  weights_manifest.json  every checkpoint tensor name/shape/dtype (the import
                         step's canonical manifest), image-path flagged

Video memory (memory_attention / memory_encoder / object pointers) is OUT OF
SCOPE and deliberately not dumped, but the weights manifest lists it.

The reference is instantiated straight from its hydra YAML by a 20-line
`_target_` walker, so no hydra/iopath/torchvision install is needed and the
config is recorded verbatim in the manifest.

Usage:
  python tools/sam2_dump_reference.py \
      --code   <sam2 repo root, the dir containing the `sam2` package> \
      --config <sam2/configs/sam2.1/sam2.1_hiera_l.yaml> \
      --ckpt   <sam2.1_hiera_large.pt> \
      --out    testdata/sam2/hiera-large [--seed 42]
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
import types

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file

# SAM 2's own preprocessing constants (sam2/utils/transforms.py).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# reference loading
# --------------------------------------------------------------------------- #
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    parent, _, leaf = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], leaf, mod)
    return mod


def stub_optional_deps():
    """The image path needs neither hydra (config composition, replaced here by
    `instantiate`) nor iopath (only the unused `Hiera(weights_path=...)` branch);
    stub them so the reference imports without extra installs."""
    if importlib.util.find_spec("iopath") is None:
        _stub("iopath")
        _stub("iopath.common")
        _stub("iopath.common.file_io", g_pathmgr=None)
    if importlib.util.find_spec("hydra") is None:
        class _GlobalHydra:
            @staticmethod
            def instance():
                return _GlobalHydra()

            @staticmethod
            def is_initialized():
                return True

        _stub("hydra", initialize_config_module=lambda *a, **k: None)
        _stub("hydra.core")
        _stub("hydra.core.global_hydra", GlobalHydra=_GlobalHydra)


def instantiate(node):
    """Minimal hydra `_target_` instantiation (no hydra dependency)."""
    if isinstance(node, dict):
        if "_target_" in node:
            mod, _, cls = node["_target_"].rpartition(".")
            kwargs = {k: instantiate(v) for k, v in node.items() if k != "_target_"}
            return getattr(importlib.import_module(mod), cls)(**kwargs)
        return {k: instantiate(v) for k, v in node.items()}
    if isinstance(node, list):
        return [instantiate(v) for v in node]
    if isinstance(node, str) and re.fullmatch(r"[-+]?\d*\.?\d+[eE][-+]?\d+", node):
        return float(node)  # YAML 1.1 leaves `1e-6` a string; omegaconf coerces it
    return node


def load_reference(code, config, ckpt):
    sys.path.insert(0, os.path.abspath(code))
    stub_optional_deps()
    with open(config) as f:
        cfg = yaml.safe_load(f)["model"]
    model = instantiate(cfg).float().eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(sd, strict=True)  # never zero-fill, never skip
    return model, cfg, sd


# --------------------------------------------------------------------------- #
# fixed synthetic input
# --------------------------------------------------------------------------- #
def synthetic_image(size, seed):
    """Deterministic RGB image in [0, 1], shape (3, size, size).

    A textured background plus two hard-edged objects (a disk and a bar) so the
    prompts below select something real and the mask logits are non-degenerate.
    """
    g = torch.Generator().manual_seed(seed)
    ys = torch.linspace(0.0, 1.0, size).view(-1, 1).expand(size, size)
    xs = torch.linspace(0.0, 1.0, size).view(1, -1).expand(size, size)
    r = 0.5 + 0.25 * torch.sin(12.0 * xs) * torch.cos(7.0 * ys)
    gg = 0.4 + 0.3 * ys
    b = 0.5 + 0.25 * torch.cos(9.0 * (xs + ys))
    img = torch.stack([r, gg, b], 0)

    # disk object
    cy, cx, rad = 0.42 * size, 0.60 * size, 0.15 * size
    yy = torch.arange(size).view(-1, 1).float()
    xx = torch.arange(size).view(1, -1).float()
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= rad * rad
    img[0][disk], img[1][disk], img[2][disk] = 0.95, 0.15, 0.10

    # bar object
    bar = torch.zeros(size, size, dtype=torch.bool)
    bar[int(0.70 * size):int(0.82 * size), int(0.12 * size):int(0.55 * size)] = True
    img[0][bar], img[1][bar], img[2][bar] = 0.10, 0.85, 0.30

    img = img + 0.02 * torch.randn(img.shape, generator=g)
    return img.clamp(0.0, 1.0).contiguous()


def normalize(img):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((img - mean) / std).unsqueeze(0).contiguous()


# --------------------------------------------------------------------------- #
# independent re-implementations used as in-dumper self-checks
# --------------------------------------------------------------------------- #
def manual_msblock(blk, x):
    """Re-derive MultiScaleBlock.forward: windowing + q_pool + naive softmax
    attention (no SDPA). Freezes the window / q_pool conventions the Rust port
    must reproduce; shares no code with the reference block."""
    shortcut = x
    y = F.layer_norm(x, (blk.dim,), blk.norm1.weight, blk.norm1.bias, 1e-6)
    if blk.dim != blk.dim_out:
        s = F.linear(y, blk.proj.weight, blk.proj.bias)
        if blk.pool is not None:
            s = F.max_pool2d(s.permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1)
        shortcut = s

    ws = blk.window_size
    b, h, w, _ = y.shape
    if ws > 0:
        assert h % ws == 0 and w % ws == 0, "test sizes must not need window padding"
        y = (y.view(b, h // ws, ws, w // ws, ws, -1)
              .permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, y.shape[-1]))
    pad_hw = (h, w)

    a = blk.attn
    bw, hh, ww, _ = y.shape
    qkv = F.linear(y, a.qkv.weight, a.qkv.bias).reshape(bw, hh * ww, 3, a.num_heads, -1)
    q, k, v = qkv.unbind(2)
    if a.q_pool is not None:
        q = q.reshape(bw, hh, ww, -1).permute(0, 3, 1, 2)
        q = F.max_pool2d(q, 2, 2).permute(0, 2, 3, 1)
        hh, ww = q.shape[1], q.shape[2]
        q = q.reshape(bw, hh * ww, a.num_heads, -1)
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    att = (qh @ kh.transpose(-1, -2)) / math.sqrt(qh.shape[-1])
    o = (att.softmax(-1) @ vh).transpose(1, 2).reshape(bw, hh, ww, -1)
    o = F.linear(o, a.proj.weight, a.proj.bias)

    if blk.q_stride:
        ws = blk.window_size // blk.q_stride[0]
        pad_hw = (shortcut.shape[1], shortcut.shape[2])
    if blk.window_size > 0:
        nb = o.shape[0] // (pad_hw[0] * pad_hw[1] // ws // ws)
        o = (o.reshape(nb, pad_hw[0] // ws, pad_hw[1] // ws, ws, ws, -1)
              .permute(0, 1, 3, 2, 4, 5).reshape(nb, pad_hw[0], pad_hw[1], -1))
    o = shortcut + o
    return o + blk.mlp(F.layer_norm(o, (blk.dim_out,), blk.norm2.weight,
                                    blk.norm2.bias, 1e-6))


def manual_fpn(neck, xs):
    """Re-derive FpnNeck.forward top-down fusion from the trunk outputs."""
    n = len(neck.convs) - 1
    out = [None] * len(neck.convs)
    prev = None
    for i in range(n, -1, -1):
        conv = neck.convs[n - i].conv
        lat = F.conv2d(xs[i], conv.weight, conv.bias)
        if i in neck.fpn_top_down_levels and prev is not None:
            prev = lat + F.interpolate(prev, scale_factor=2.0, mode=neck.fpn_interp_model)
        else:
            prev = lat
        out[i] = prev
    return out


# --------------------------------------------------------------------------- #
# dumping helpers
# --------------------------------------------------------------------------- #
class Dump:
    def __init__(self, out):
        self.out = out
        self.files = {}
        os.makedirs(out, exist_ok=True)

    def save(self, name, tensors):
        # everything as f32 — brain's safetensors reader is F32/F16/BF16-only,
        # and point labels / indices are exactly representable.
        t = {k: v.detach().to(torch.float32).contiguous().clone()
             for k, v in tensors.items()}
        path = os.path.join(self.out, name)
        save_file(t, path)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        self.files[name] = {
            "sha256": sha,
            "bytes": os.path.getsize(path),
            "tensors": {k: {"shape": list(v.shape), "dtype": "float32"}
                        for k, v in t.items()},
        }
        print(f"  wrote {name} ({os.path.getsize(path) / 1e6:.1f} MB, "
              f"{len(t)} tensors)", flush=True)


def compare(checks, name, got, ref):
    """Record max-abs / relative / cosine of `got` vs the reference `ref`."""
    got, ref = got.double().flatten(), ref.double().flatten()
    scale = ref.abs().max().item()
    max_abs = (got - ref).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
    checks[name] = {"max_abs": max_abs,
                    "rel": max_abs / scale if scale > 0 else max_abs,
                    "cosine": cos}
    return checks[name]["rel"]


def hook_out(store, key, transform=None):
    def fn(_m, _a, out):
        store[key] = out if transform is None else transform(out)
    return fn


def hook_in(store, key):
    def fn(_m, args):
        store[key] = args[0].clone()
    return fn


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="sam2 repo root (contains the sam2 package)")
    ap.add_argument("--config", required=True, help="sam2.1_hiera_*.yaml")
    ap.add_argument("--ckpt", required=True, help="sam2.1_hiera_*.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    print("loading reference (CPU, fp32) ...", flush=True)
    model, cfg, sd = load_reference(args.code, args.config, args.ckpt)
    trunk, neck = model.image_encoder.trunk, model.image_encoder.neck
    size = model.image_size
    dump = Dump(args.out)
    checks = {}

    # ---- weights manifest (free architecture doc for the import step) -------
    video_only = ("memory_attention.", "memory_encoder.", "maskmem_tpos_enc",
                  "no_mem_embed", "no_mem_pos_enc", "mask_downsample.",
                  "obj_ptr_tpos_proj.", "no_obj_embed_spatial")
    wm = {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", ""),
              "image_path": not k.startswith(video_only)}
          for k, v in sd.items()}
    with open(os.path.join(args.out, "weights_manifest.json"), "w") as f:
        json.dump({"count": len(wm),
                   "image_path_count": sum(1 for v in wm.values() if v["image_path"]),
                   "tensors": wm}, f, indent=1)
    print(f"  weights_manifest.json: {len(wm)} tensors, "
          f"{sum(1 for v in wm.values() if v['image_path'])} on the image path", flush=True)

    # ---- input --------------------------------------------------------------
    raw = synthetic_image(size, args.seed)
    inp = normalize(raw)
    dump.save("input.safetensors", {
        "image_rgb01": raw,
        "image_norm": inp,
        "pixel_mean": torch.tensor(IMAGENET_MEAN),
        "pixel_std": torch.tensor(IMAGENET_STD),
    })

    # ---- trunk / neck taps via forward hooks --------------------------------
    depth = len(trunk.blocks)
    taps = sorted({0, depth - 1, *trunk.stage_ends, *trunk.q_pool_blocks,
                   *(trunk.global_att_blocks or ())})
    check_blocks = sorted({0, trunk.q_pool_blocks[0],
                           (trunk.global_att_blocks or (0,))[0]})
    store, handles = {}, []
    handles.append(trunk.patch_embed.register_forward_hook(
        hook_out(store, "patch_embed", lambda o: o.clone())))
    for i in taps:
        handles.append(trunk.blocks[i].register_forward_hook(
            hook_out(store, f"blk{i:02d}_out", lambda o: o.clone())))
    for i in check_blocks:
        handles.append(trunk.blocks[i].register_forward_pre_hook(
            hook_in(store, f"blk{i:02d}_in")))
    for i in range(len(neck.convs)):
        handles.append(neck.convs[i].register_forward_hook(
            hook_out(store, f"neck_conv{i}", lambda o: o.clone())))
    handles.append(trunk.register_forward_hook(
        hook_out(store, "trunk_out", lambda o: [t.clone() for t in o])))
    handles.append(neck.register_forward_hook(
        hook_out(store, "neck_out", lambda o: ([t.clone() for t in o[0]],
                                               [t.clone() for t in o[1]]))))

    print("forward: image encoder ...", flush=True)
    backbone_out = model.forward_image(inp)
    for h in handles:
        h.remove()

    # self-check: block input == patch_embed + interpolated pos embed
    pos_embed = trunk._get_pos_embed(store["patch_embed"].shape[1:3])
    compare(checks, "pos_embed_plus_patch_vs_block0_in",
            store["patch_embed"] + pos_embed, store["blk00_in"])

    # self-check: independently re-derived windowed MHA + q_pool blocks
    for i in check_blocks:
        man = manual_msblock(trunk.blocks[i], store[f"blk{i:02d}_in"])
        compare(checks, f"manual_block{i:02d}", man, store[f"blk{i:02d}_out"])

    trunk_feats = store["trunk_out"]
    fpn_out, fpn_pos = store["neck_out"]
    for i, (a, b) in enumerate(zip(manual_fpn(neck, trunk_feats), fpn_out)):
        compare(checks, f"manual_fpn_level{i}", a, b)

    tr = {"pos_embed_interp": pos_embed, "patch_embed": store["patch_embed"]}
    for i in taps:
        tr[f"blk{i:02d}_out"] = store[f"blk{i:02d}_out"]
    for i in check_blocks:
        tr[f"blk{i:02d}_in"] = store[f"blk{i:02d}_in"]
    for i, t in enumerate(trunk_feats):
        tr[f"trunk_feat{i}"] = t
    dump.save("trunk.safetensors", tr)

    nk = {}
    # convs[n - i] is applied to trunk level i — record it under the LEVEL index
    n = len(neck.convs) - 1
    for i in range(len(neck.convs)):
        nk[f"lateral_level{n - i}"] = store[f"neck_conv{i}"]
    for i, (o, p) in enumerate(zip(fpn_out, fpn_pos)):
        nk[f"fpn_level{i}"] = o
        nk[f"possine_level{i}"] = p

    # post-scalp features, conv_s0/conv_s1 projections and the SAM image embedding
    _, vision_feats, vision_pos, feat_sizes = model._prepare_backbone_features(backbone_out)
    for i, f in enumerate(backbone_out["backbone_fpn"]):
        nk[f"backbone_fpn{i}"] = f          # level 0/1 already conv_s0/conv_s1 projected
    # SAM2ImagePredictor.set_image: the lowest-resolution level carries the
    # no-memory embedding on the image path (video memory is out of scope).
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    feats = [f.permute(1, 2, 0).view(1, -1, *sz)
             for f, sz in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
    image_embed, high_res_feats = feats[-1], feats[:-1]
    nk["no_mem_embed"] = model.no_mem_embed
    nk["image_embed"] = image_embed
    for i, f in enumerate(high_res_feats):
        nk[f"high_res_feat{i}"] = f
    dump.save("neck.safetensors", nk)

    # ---- prompt cases -------------------------------------------------------
    dense_pe = model.sam_prompt_encoder.get_dense_pe()
    cy, cx = 0.42 * size, 0.60 * size          # disk centre (y, x)
    by0, by1 = 0.70 * size, 0.82 * size        # bar rows
    bx0, bx1 = 0.12 * size, 0.55 * size        # bar cols
    cases = {
        "point1": dict(coords=[[cx, cy]], labels=[1], multimask=True),
        "point1_single": dict(coords=[[cx, cy]], labels=[1], multimask=False),
        "point2_negpos": dict(coords=[[cx, cy], [0.2 * size, 0.2 * size]],
                              labels=[1, 0], multimask=True),
        "box_bar": dict(coords=[[bx0, by0], [bx1, by1]], labels=[2, 3], multimask=True),
        "point_and_mask": dict(coords=[[0.33 * size, 0.76 * size]], labels=[1],
                               multimask=False, mask=True),
    }

    for name, spec in cases.items():
        print(f"forward: prompt case {name} ...", flush=True)
        coords = torch.tensor([spec["coords"]], dtype=torch.float32)
        labels = torch.tensor([spec["labels"]], dtype=torch.int32)

        mask_hi = sam_mask_prompt = None
        if spec.get("mask"):
            # a coarse box-shaped mask logit prompt over the bar object, at image
            # resolution: exercises the reference's antialiased 1024 -> 256
            # downsample. The resulting `sam_mask_prompt` is dumped so a port can
            # replay the decoder without an antialias kernel.
            mask_hi = torch.full((1, 1, size, size), -8.0)
            mask_hi[..., int(by0):int(by1), int(bx0):int(bx1)] = 8.0
            sam_mask_prompt = F.interpolate(
                mask_hi, size=model.sam_prompt_encoder.mask_input_size,
                align_corners=False, mode="bilinear", antialias=True)

        st, hs_handles = {}, []
        tf = model.sam_mask_decoder.transformer
        for li, layer in enumerate(tf.layers):
            hs_handles.append(layer.register_forward_hook(
                hook_out(st, f"twoway{li}", lambda o: (o[0].clone(), o[1].clone()))))
        hs_handles.append(tf.final_attn_token_to_image.register_forward_hook(
            hook_out(st, "final_attn", lambda o: o.clone())))
        hs_handles.append(model.sam_prompt_encoder.register_forward_hook(
            hook_out(st, "prompt", lambda o: (o[0].clone(), o[1].clone()))))
        up = model.sam_mask_decoder.output_upscaling
        hs_handles.append(up[0].register_forward_hook(
            hook_out(st, "dc1", lambda o: o.clone())))
        hs_handles.append(up[3].register_forward_hook(
            hook_out(st, "dc2", lambda o: o.clone())))

        (low_res_multimasks, high_res_multimasks, ious, low_res_masks,
         high_res_masks, obj_ptr, object_score_logits) = model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs={"point_coords": coords, "point_labels": labels},
            mask_inputs=mask_hi,
            high_res_features=high_res_feats,
            multimask_output=spec["multimask"])
        for h in hs_handles:
            h.remove()

        sparse, dense = st["prompt"]
        # re-run the decoder's internals to tap the intermediates the hooks
        # cannot see (tokens, hs/src split, upscaled embedding, hyper_in)
        dec = model.sam_mask_decoder
        output_tokens = torch.cat([dec.obj_score_token.weight, dec.iou_token.weight,
                                   dec.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse), dim=1)
        src = image_embed + dense
        pos_src = dense_pe.expand(tokens.shape[0], -1, -1, -1)
        hs, src_out = dec.transformer(src, pos_src, tokens)
        b, c, h, w = src.shape
        src_img = src_out.transpose(1, 2).view(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = dec.output_upscaling
        upscaled = act1(ln1(dc1(src_img) + high_res_feats[1]))
        upscaled = act2(dc2(upscaled) + high_res_feats[0])
        # obj_score token is index 0, IoU token index 1 (pred_obj_scores => s=1)
        mask_tokens_out = hs[:, 2:2 + dec.num_mask_tokens, :]
        hyper_in = torch.stack([dec.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                                for i in range(dec.num_mask_tokens)], dim=1)
        ub, uc, uh, uw = upscaled.shape
        masks_all = (hyper_in @ upscaled.view(ub, uc, uh * uw)).view(ub, -1, uh, uw)
        iou_all = dec.iou_prediction_head(hs[:, 1, :])
        obj_score = dec.pred_obj_score_head(hs[:, 0, :])

        # self-check: the replayed decoder reproduces the model's own outputs
        sel = slice(1, None) if spec["multimask"] else slice(0, 1)
        masks_sel = torch.where((obj_score > 0)[:, None, None],
                                masks_all[:, sel], -1024.0)
        compare(checks, f"{name}_replay_masks", masks_sel, low_res_multimasks)
        compare(checks, f"{name}_replay_iou", iou_all[:, sel], ious)
        compare(checks, f"{name}_replay_objscore", obj_score, object_score_logits)
        # self-check: hi-res masks are the low-res masks bilinearly upsampled
        compare(checks, f"{name}_hires_upsample",
                F.interpolate(low_res_multimasks.float(), size=(size, size),
                              mode="bilinear", align_corners=False),
                high_res_multimasks)

        t = {
            "point_coords": coords, "point_labels": labels.float(),
            "multimask_output": torch.tensor([1.0 if spec["multimask"] else 0.0]),
            "sparse_embeddings": sparse, "dense_embeddings": dense,
            "dense_pe": dense_pe,
            "tokens": tokens, "src_in": src,
            "twoway0_queries": st["twoway0"][0], "twoway0_keys": st["twoway0"][1],
            "twoway1_queries": st["twoway1"][0], "twoway1_keys": st["twoway1"][1],
            "final_attn_out": st["final_attn"],
            "hs": hs, "src_out": src_out,
            "dc1_out": st["dc1"], "dc2_out": st["dc2"],
            "upscaled_embedding": upscaled, "hyper_in": hyper_in,
            "masks_all4": masks_all, "iou_all4": iou_all,
            "object_score_logits": object_score_logits,
            "low_res_multimasks": low_res_multimasks,
            "high_res_multimasks": high_res_multimasks,
            "ious": ious,
            "low_res_masks_best": low_res_masks,
            "high_res_masks_best": high_res_masks,
            "obj_ptr": obj_ptr,
            "best_iou_index": torch.argmax(ious, dim=-1).float(),
            "mask_positive_fraction": (high_res_masks > 0).float().mean().view(1),
        }
        if sam_mask_prompt is not None:
            t["mask_input_hires"] = mask_hi
            t["mask_input_lowres"] = sam_mask_prompt
        dump.save(f"case_{name}.safetensors", t)

    # ---- manifest -----------------------------------------------------------
    for k, v in sorted(checks.items()):
        print(f"  check {k}: max_abs {v['max_abs']:.3e} rel {v['rel']:.3e} "
              f"cos {v['cosine']:.8f}", flush=True)
    worst = max(v["rel"] for v in checks.values())
    assert worst < 1e-4, f"self-check failed (worst rel {worst:.3e}): {checks}"

    # per-block table, with the spatial size each block sees and whether the
    # reference's window_partition has to zero-pad at this resolution.
    blocks, h = [], size // 4
    for i, b in enumerate(trunk.blocks):
        ho = h // 2 if b.q_stride else h
        blocks.append({"index": i, "dim": b.dim, "dim_out": b.dim_out,
                       "num_heads": b.attn.num_heads, "window_size": b.window_size,
                       "q_pool": b.q_stride is not None,
                       "in_hw": [h, h], "out_hw": [ho, ho],
                       "window_pad": bool(b.window_size and h % b.window_size)})
        h = ho
    manifest = {
        "model": cfg,
        "params": {
            "seed": args.seed,
            "image_size": size,
            "backbone_stride": model.backbone_stride,
            "pixel_mean": list(IMAGENET_MEAN),
            "pixel_std": list(IMAGENET_STD),
            "num_feature_levels": model.num_feature_levels,
            "use_high_res_features_in_sam": model.use_high_res_features_in_sam,
            "multimask_output_in_sam": model.multimask_output_in_sam,
            "iou_prediction_use_sigmoid": model.iou_prediction_use_sigmoid,
            "pred_obj_scores": model.pred_obj_scores,
            "dynamic_multimask_via_stability":
                model.sam_mask_decoder.dynamic_multimask_via_stability,
            "num_mask_tokens": model.sam_mask_decoder.num_mask_tokens,
            "mask_input_size": list(model.sam_prompt_encoder.mask_input_size),
            "image_embedding_size": list(model.sam_prompt_encoder.image_embedding_size),
            "feat_sizes": [list(s) for s in feat_sizes],
            "trunk_stage_ends": trunk.stage_ends,
            "trunk_q_pool_blocks": trunk.q_pool_blocks,
            "trunk_global_att_blocks": list(trunk.global_att_blocks or ()),
            "trunk_window_spec": list(trunk.window_spec),
            "trunk_channel_list": trunk.channel_list,
            "tapped_blocks": taps,
            "manually_rechecked_blocks": check_blocks,
            "prompt_cases": {k: {kk: vv for kk, vv in v.items()} for k, v in cases.items()},
            "checkpoint": os.path.basename(args.ckpt),
            "config": os.path.basename(args.config),
        },
        "trunk_blocks": blocks,
        "self_checks_max_abs": checks,
        "files": dump.files,
        "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
