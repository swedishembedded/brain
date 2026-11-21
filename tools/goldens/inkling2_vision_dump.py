#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Bake Inkling 2 vision goldens (image preprocessor and vision tower).

Three dumps, run by hand with the pinned reference environment:

  preproc  inkling2_vision_dump.py preproc [out_dir]
           A handful of PNG test images (RGB, RGBA with every alpha class, odd
           sizes, 1x1, an exact multiple of the region, grayscale) and the
           reference processor's `pixel_values` / `image_grid_thw` for each,
           read back from the PNG with PIL exactly as a caller would.
           The pinned environment has no torchvision, so the reference
           `Inkling2ImageProcessor` cannot be imported; `reference_preprocess`
           below replays its `convert_to_rgb` and `_preprocess` step for step on
           torch tensors (integer checkerboard composite, float pad of -1 on the
           0-255 scale, torchvision's `normalize` = `sub` then `div`, the fused
           mean/std of `rescale_and_normalize`).

  tiny     inkling2_vision_dump.py tiny [out_dir]
           A random small tower (hidden 64, 4 heads, 3 layers, 8x8 position
           table) with every parameter seeded to something non-trivial, written
           as a checkpoint IN THE PUBLISHED NAMING (`model.visual.*`, f32), and
           the activations of one forward over three images: the embeddings
           (class token rows included), the pre-LayerNorm, every layer, the last
           hidden state, the merger input and output, the interpolation taps and
           weights, the rope position ids and cos/sin tables.

  real     inkling2_vision_dump.py real <checkpoint dir> [out_dir]
           The published checkpoint's vision tensors (bf16 widened to f32, which
           is exact) through the reference tower on CPU in f32 for two small
           images; records the embeddings, layers 0 / 11 / 22, the last hidden
           state and the merger output.

Every tensor is f32 (integers cast exactly). `manifest.json` carries shapes, the
run parameters and library versions.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from golden_source import source_block  # noqa: E402

SEED = 20261122
TESTDATA = os.path.join(HERE, "..", "..", "testdata", "golden", "inkling2", "vision")
REGION = 40  # processor size {"height": 40, "width": 40}
RESCALE = 0.00392156862745098
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
PAD_FILL = -1.0  # on the 0-255 scale, before rescale and normalise


# ----------------------------------------------------------------------------
# Reference processor (torchvision-free replay of Inkling2ImageProcessor)
# ----------------------------------------------------------------------------
def convert_to_rgb(image):
    """`Inkling2ImageProcessor.convert_to_rgb` on a uint8 `[C, H, W]` tensor."""
    if image.shape[-3] == 1:  # tvF.grayscale_to_rgb
        image = image.expand(3, -1, -1)
    if image.shape[-3] != 4:
        return image
    height, width = image.shape[-2:]
    rows = torch.arange(height)[:, None] // 8
    columns = torch.arange(width)[None, :] // 8
    background = torch.where((rows + columns) % 2 == 0, 204, 153)
    rgba = image.to(torch.int32)
    alpha = rgba[3:4]
    return ((rgba[:3] * alpha + background * (255 - alpha) + 127) // 255).to(torch.uint8)


def reference_preprocess(pil_image):
    """`pixel_values [tokens, 40, 40, 3]` and `[1, rows, cols]` for one PIL image."""
    mode = "RGBA" if pil_image.has_transparency_data else "RGB"
    image = torch.from_numpy(np.asarray(pil_image.convert(mode)).copy()).permute(2, 0, 1)
    image = convert_to_rgb(image)
    channels, height, width = image.shape
    rows, columns = math.ceil(height / REGION), math.ceil(width / REGION)
    image = image.float()
    # tvF.pad(fill=-1.0, padding_mode="constant") on a float tensor
    image = F.pad(image, (0, columns * REGION - width, 0, rows * REGION - height), value=PAD_FILL)
    # rescale_and_normalize with do_rescale and do_normalize: fused mean/std
    mean = torch.tensor(MEAN) * (1.0 / RESCALE)
    std = torch.tensor(STD) * (1.0 / RESCALE)
    image = image.sub(mean.view(-1, 1, 1)).div_(std.view(-1, 1, 1))
    image = image.reshape(channels, rows, REGION, columns, REGION).permute(1, 3, 2, 4, 0)
    return image.flatten(0, 1).contiguous(), [1, rows, columns]


def test_images():
    """name -> PIL image. Structured plus seeded noise so no two cells agree."""
    rs = np.random.RandomState(SEED)

    def texture(h, w):
        yy, xx = np.mgrid[0:h, 0:w]
        base = np.stack([(xx * 7 + yy * 3) % 256, (xx * 2 + yy * 11) % 256, (xx * yy) % 256], -1)
        noise = rs.randint(0, 64, size=(h, w, 3))
        return ((base + noise) % 256).astype(np.uint8)

    out = {}
    out["rgb_odd"] = Image.fromarray(texture(61, 83), "RGB")
    rgba = np.concatenate([texture(45, 70), np.zeros((45, 70, 1), np.uint8)], -1)
    classes = np.array([0, 1, 2, 126, 127, 128, 129, 253, 254, 255], np.uint8)
    rgba[..., 3] = classes[rs.randint(0, len(classes), size=(45, 70))]
    rgba[:8, :8, 3] = 0  # one full transparent checkerboard cell
    rgba[8:16, 8:16, 3] = 255  # one full opaque cell
    out["rgba_mixed"] = Image.fromarray(rgba, "RGBA")
    out["rgb_1x1"] = Image.fromarray(np.array([[[200, 17, 99]]], np.uint8), "RGB")
    out["rgba_1x1"] = Image.fromarray(np.array([[[200, 17, 99, 100]]], np.uint8), "RGBA")
    out["rgb_exact"] = Image.fromarray(texture(40, 80), "RGB")
    out["gray_odd"] = Image.fromarray(texture(47, 33)[..., 0], "L")
    opaque = np.concatenate([texture(41, 41), np.full((41, 41, 1), 255, np.uint8)], -1)
    out["rgba_opaque"] = Image.fromarray(opaque, "RGBA")
    return out


def dump_preproc(out):
    d = os.path.join(out, "preproc")
    os.makedirs(d, exist_ok=True)
    tensors, manifest = {}, {}
    for name, img in test_images().items():
        path = os.path.join(d, f"{name}.png")
        img.save(path)
        pv, grid = reference_preprocess(Image.open(path))  # the caller's route: decode the file
        tensors[f"{name}.pixels"] = pv
        tensors[f"{name}.grid"] = torch.tensor([grid], dtype=torch.float32)
        manifest[name] = {"size": list(img.size), "mode": img.mode, "grid": grid, "pixels": list(pv.shape)}
    save_file(tensors, os.path.join(d, "golden.safetensors"))
    json.dump({"cases": manifest, "torch": torch.__version__, "pillow": __import__("PIL").__version__,
               "note": "torchvision absent from the pinned venv; processor steps replayed on torch tensors"},
              open(os.path.join(d, "manifest.json"), "w"), indent=1, sort_keys=True)
    print("preproc ->", d, sorted(manifest))


# ----------------------------------------------------------------------------
# Vision tower
# ----------------------------------------------------------------------------
def published_name(hf):
    """Reference parameter name (`Inkling2VisionModel`) -> published checkpoint name."""
    p = "model.visual."
    if hf.startswith("embeddings."):
        rest = hf[len("embeddings."):]
        return p + "backbone.embeddings." + ("position_embedding" if rest == "position_embedding.weight" else rest)
    if hf.startswith("pre_layernorm."):
        return p + "backbone.ln_pre." + hf[len("pre_layernorm."):]
    if hf.startswith("merger.linear_1."):
        return p + "proj_in." + hf[len("merger.linear_1."):]
    if hf.startswith("merger.linear_2."):
        return p + "proj_out." + hf[len("merger.linear_2."):]
    if hf.startswith("layers."):
        _, n, rest = hf.split(".", 2)
        for a, b in [("layernorm_before.", "ln_1."), ("layernorm_after.", "ln_2."), ("mlp.fc1.", "mlp.c_fc."),
                     ("mlp.fc2.", "mlp.c_proj."), ("lambda_1", "ls_1.gamma"), ("lambda_2", "ls_2.gamma")]:
            if rest.startswith(a):
                rest = b + rest[len(a):]
                break
        return f"{p}backbone.layers.{n}.{rest}"
    raise KeyError(hf)


def vision_config(tiny):
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
    rope = {"rope_type": "axial", "rope_theta": 10000.0}
    if tiny:
        return Inkling2VisionConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, posemb_grid_size=8,
                                    projection_hidden_dim=96, text_hidden_size=48, rope_parameters=rope)
    return Inkling2VisionConfig(rope_parameters=rope, text_hidden_size=2304)


def build_model(cfg):
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
    cfg._attn_implementation = "eager"
    return Inkling2VisionModel(cfg).float().eval()


def randomise(model):
    """Every parameter non-trivial: LayerNorm gains near 1, layer scales and class tokens random."""
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("lambda_1") or name.endswith("lambda_2"):
                p.copy_(0.05 + 0.3 * torch.rand(p.shape, generator=g))
            elif "layernorm" in name:
                gain = name.endswith("weight")
                p.copy_((1.0 + 0.2 * torch.randn(p.shape, generator=g)) if gain else 0.1 * torch.randn(p.shape, generator=g))
            else:
                fan_in = p.shape[-1] if p.ndim > 1 else 16
                p.copy_(torch.randn(p.shape, generator=g) * (0.5 if p.ndim == 1 else 1.0 / math.sqrt(fan_in)))


def run_tower(model, images, taps):
    """Forward over PIL images; returns the activation dict and the image grid."""
    pvs, grids = zip(*[reference_preprocess(i) for i in images])
    pixel_values = torch.cat(pvs)
    grid = torch.tensor(grids)
    out = {"pixel_values": pixel_values, "image_grid_thw": grid.float()}
    hooks = []

    def grab(key):
        def hook(_m, _i, o):
            out[key] = (o[0] if isinstance(o, tuple) else o).detach().clone().float()
        return hook

    hooks.append(model.embeddings.register_forward_hook(grab("embeddings")))
    hooks.append(model.pre_layernorm.register_forward_hook(grab("pre_layernorm")))
    hooks.append(model.rotary_emb.register_forward_hook(lambda _m, _i, o: out.update(rope_cos=o[0].clone(), rope_sin=o[1].clone())))
    for i, layer in enumerate(model.layers):
        if i in taps:
            hooks.append(layer.register_forward_hook(grab(f"layer.{i}")))
    hooks.append(model.merger.register_forward_hook(lambda _m, i, _o: out.update(merger_in=i[0].detach().clone())))
    with torch.no_grad():
        res = model(pixel_values=pixel_values, image_grid_thw=grid)
    for h in hooks:
        h.remove()
    out["last_hidden"] = res.last_hidden_state.float()
    out["features"] = res.pooler_output.float()
    return out, grid


def dump_tower_positions(model, grid, out):
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    merge = model.config.spatial_merge_size
    patch_grid = grid * torch.tensor([1, merge, merge])
    idx, w = get_vision_interpolation_indices_and_weights(
        patch_grid, num_grid_per_side=model.config.posemb_grid_size, mode="bilinear", align_corners=False,
        spatial_merge_size=merge)
    out["interp_indices"] = idx.float()
    out["interp_weights"] = w.float()
    out["position_ids"] = get_vision_position_ids(patch_grid, merge).float()


def write_dump(out, tensors, manifest):
    os.makedirs(out, exist_ok=True)
    save_file({k: v.contiguous().float() for k, v in tensors.items()}, os.path.join(out, "golden.safetensors"))
    manifest["shapes"] = {k: list(v.shape) for k, v in tensors.items()}
    manifest["torch"] = torch.__version__
    import transformers
    manifest["transformers"] = transformers.__version__
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1, sort_keys=True)


def arch_block(cfg):
    return {"hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads, "posemb_grid_size": cfg.posemb_grid_size,
            "text_hidden_size": cfg.text_hidden_size}


def dump_tiny(out):
    out = os.path.join(out, "tiny")
    os.makedirs(out, exist_ok=True)
    cfg = vision_config(tiny=True)
    torch.manual_seed(SEED)
    model = build_model(cfg)
    randomise(model)
    published = {published_name(k): v.detach().clone().float() for k, v in model.state_dict().items()}
    save_file(published, os.path.join(out, "model.safetensors"))
    # the published naming round-trips: a fresh model loaded from it agrees with the dumped one
    inverse = {published_name(k): k for k in model.state_dict()}
    fresh = build_model(vision_config(tiny=True))
    fresh.load_state_dict({inverse[k]: v for k, v in load_file(os.path.join(out, "model.safetensors")).items()}, strict=True)
    imgs_all = test_images()
    images = [imgs_all["rgb_odd"].crop((0, 0, 70, 50)), imgs_all["rgba_mixed"].crop((0, 0, 70, 45)).resize((90, 45)), imgs_all["rgb_1x1"]]
    taps = set(range(cfg.num_hidden_layers))
    acts, grid = run_tower(model, images, taps)
    again, _ = run_tower(fresh, images, taps)
    assert torch.equal(acts["features"], again["features"]), "published naming does not round-trip"
    dump_tower_positions(model, grid, acts)
    manifest = {
        "config": {"hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers,
                   "num_attention_heads": cfg.num_attention_heads, "intermediate_size": cfg.intermediate_size,
                   "posemb_grid_size": cfg.posemb_grid_size, "projection_hidden_dim": cfg.projection_hidden_dim,
                   "text_hidden_size": cfg.text_hidden_size, "rope_theta": 10000.0, "layer_norm_eps": cfg.layer_norm_eps,
                   "patch_size": cfg.patch_size, "spatial_merge_size": cfg.spatial_merge_size},
        "grid": grid.tolist(),
        "source": source_block(checkpoint="inkling2-vision-tiny-random", files=[os.path.join(out, "model.safetensors")], identity=arch_block(cfg)),
    }
    write_dump(out, acts, manifest)
    print("tiny ->", out, "features", tuple(acts["features"].shape))


def dump_real(ckpt, out):
    out = os.path.join(out, "real")
    cfg = vision_config(tiny=False)
    model = build_model(cfg)
    f = os.path.join(ckpt, "model-00001-of-00001.safetensors")
    from safetensors import safe_open
    inverse = {published_name(k): k for k in model.state_dict()}
    state = {}
    with safe_open(f, "pt") as sf:
        for k in sf.keys():
            if k.startswith("model.visual."):
                state[inverse[k]] = sf.get_tensor(k).float()
    missing = set(model.state_dict()) - set(state)
    assert not missing, f"checkpoint lacks vision tensors {sorted(missing)[:5]}"
    model.load_state_dict(state, strict=True)
    imgs = test_images()
    images = [imgs["rgb_odd"].crop((0, 0, 60, 40)), imgs["rgba_mixed"].crop((0, 0, 40, 70))]
    acts, grid = run_tower(model, images, {0, 11, 22})
    dump_tower_positions(model, grid, acts)
    manifest = {"grid": grid.tolist(), "source": source_block(
        checkpoint="thinkingmachines/Inkling-2-Mini", files=[f], identity=arch_block(cfg), hash_files=False)}
    write_dump(out, acts, manifest)
    print("real ->", out, "features", tuple(acts["features"].shape))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "preproc":
        dump_preproc(sys.argv[2] if len(sys.argv) > 2 else TESTDATA)
    elif mode == "tiny":
        dump_tiny(sys.argv[2] if len(sys.argv) > 2 else TESTDATA)
    elif mode == "real" and len(sys.argv) > 2:
        dump_real(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else TESTDATA)
    else:
        sys.exit(__doc__)
