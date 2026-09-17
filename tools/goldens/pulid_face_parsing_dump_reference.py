#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump PuLID-FLUX's real face-preprocessing reference goldens for brain's
`crates/pulid` parity tests: facexlib's BiSeNet face parsing plus PuLID's own
background-whiten / face-grayscale masking, transcribed verbatim from
`pulid/pipeline_flux.py::PuLIDPipeline.get_id_embedding` in the official
ToTheBeginning/PuLID repo (v0.9.1).

This is the ONE documented numeric divergence between brain's served PuLID
path and upstream: brain currently resizes the aligned face crop straight to
EVA-CLIP-L/336 (`crates/pulid/src/caps.rs`'s module docs), instead of
reproducing this alignment -> BiSeNet parse -> whiten/gray -> bicubic resize
chain. This dumper makes that reproducible and gate-able.

**Landmarks are supplied by the CALLER (`--kps`), not detected here.** This
isolates exactly what this dumper is meant to gate - the alignment-warp math,
BiSeNet's own forward pass, and the masking/resize logic - from face
DETECTION, which is a separate, already-parity-gated concern
(`crates/scrfd`). Feed it brain's own `brain scrfd detect --json` output on
the same photo and the golden and a brain run start from the identical
5-point input.

Four files:

  bisenet.safetensors      BiSeNet(num_class=19) forward on the aligned
                           512x512 crop (imagenet-normalized per the
                           reference), tapped at every named submodule
                           output - the layer-by-layer ladder for a
                           stage-by-stage parity test, not just end-to-end.
  align.safetensors        The 512x512 aligned crop itself (`align_face`,
                           [0,1] RGB) and the affine matrix, so the WARP can
                           be gated independently of BiSeNet.
  masked.safetensors       The full chain's OUTPUT: the background-whitened /
                           face-grayscaled image, and its bicubic resize to
                           EVA-CLIP-L/336 - the exact tensor that should
                           replace today's plain resize before EVA-CLIP
                           normalization (which is unchanged and already
                           parity-gated elsewhere in this crate).
  parsing_bisenet.safetensors  facexlib's own `parsing_bisenet.pth` weights,
                           re-serialized (NOT re-derived - the same
                           state_dict `init_parsing_model` loaded above, just
                           written to a format `checkpoint::safetensors`
                           reads) because the released `.pth` is torch's
                           pre-1.6 legacy pickle format, which
                           `checkpoint::torchpt` does not read - see
                           `crates/bisenet/src/import.rs`'s module docs.

Usage:
  python tools/goldens/pulid_face_parsing_dump_reference.py \
      --photo /path/to/photo.jpg \
      --kps x1,y1,x2,y2,x3,y3,x4,y4,x5,y5 \
      --testdata testdata

`--kps` is the flat 10-float SCRFD 5-point output (left eye, right eye, nose,
left mouth, right mouth - `brain scrfd detect --json`'s own `kps` field,
flattened).
"""

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facexlib.parsing import init_parsing_model
from safetensors.torch import save_file
from torchvision.transforms.functional import normalize, resize
from torchvision.transforms import InterpolationMode

# The exact standard 5-landmark FFHQ-512 destination template
# (facexlib.utils.face_restoration_helper.FaceRestoreHelper, template_3points=False,
# face_size=512, crop_ratio=(1,1) - PuLID's own FaceRestoreHelper init args).
FFHQ_DST_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float64,
)

# PuLID pipeline_flux.py::get_id_embedding, verbatim.
BG_LABELS = [0, 16, 18, 7, 8, 9, 14, 15]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EVA_L336_SIZE = 336
# EVA-CLIP-L/336's own published normalization (OPENAI_DATASET_MEAN/STD, the
# facexlib/PuLID fallback when the vision tower exposes no image_mean/std of
# its own - `clip::EvaVisionConfig::eva02_l336` uses the same constants
# elsewhere in this workspace).
EVA_MEAN = [0.48145466, 0.4578275, 0.40821073]
EVA_STD = [0.26862954, 0.26130258, 0.27577711]


def to_gray(img: torch.Tensor) -> torch.Tensor:
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return x.repeat(1, 3, 1, 1)


def save(out_dir, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out_dir, name)
    save_file(tensors, path)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifest[name] = {"sha256": sha, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    print(f"wrote {path} ({len(tensors)} tensors)", flush=True)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402  (tools/goldens is this file's own dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--kps", required=True, help="flat x1,y1,...,x5,y5 (SCRFD 5-point order)")
    ap.add_argument("--testdata", default="testdata")
    args = ap.parse_args()

    kps = np.array([float(v) for v in args.kps.split(",")], dtype=np.float64).reshape(5, 2)

    img_bgr = cv2.imread(args.photo)
    if img_bgr is None:
        raise SystemExit(f"could not read {args.photo}")

    # --- alignment: identical formula to FaceRestoreHelper.align_warp_face ---
    affine_matrix, _ = cv2.estimateAffinePartial2D(kps, FFHQ_DST_512, method=cv2.LMEDS)
    align_face_bgr = cv2.warpAffine(img_bgr, affine_matrix, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
    align_face_rgb = cv2.cvtColor(align_face_bgr, cv2.COLOR_BGR2RGB)

    out_dir = os.path.join(args.testdata, "pulid")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}

    input_t = torch.from_numpy(align_face_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)  # [1,3,512,512] RGB [0,1]

    save(
        out_dir,
        "align.safetensors",
        {
            "align_face_rgb": input_t[0],
            "affine_matrix": torch.from_numpy(affine_matrix),
            "kps_src": torch.from_numpy(kps),
            "dst_template": torch.from_numpy(FFHQ_DST_512),
        },
        manifest,
    )

    # --- BiSeNet forward, tapped ---
    model = init_parsing_model(model_name="bisenet", device="cpu")
    model.eval()

    # facexlib's released `parsing_bisenet.pth` is torch's pre-1.6 LEGACY
    # format (a bare PROTO-2 pickle stream, no zip container) - the ONE
    # checkpoint in this workspace's import path that isn't either a
    # zip-container `.pt`/`.pth` (`checkpoint::torchpt`) or `.safetensors`
    # (`checkpoint::safetensors`). Convert it once, here, alongside the
    # goldens it feeds, so `crates/bisenet::import::read` has a format its
    # reader actually supports - a re-serialization of the SAME state_dict
    # already loaded above, not a re-derivation.
    from safetensors.torch import save_file as _save_bisenet_weights

    _bisenet_out = {k: v.contiguous() for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    _bisenet_weights_path = os.path.join(out_dir, "parsing_bisenet.safetensors")
    _save_bisenet_weights(_bisenet_out, _bisenet_weights_path)
    print(f"wrote {_bisenet_weights_path} ({len(_bisenet_out)} tensors)", flush=True)

    taps = {}
    hooks = []
    tap_names = [
        "cp.resnet.layer1", "cp.resnet.layer2", "cp.resnet.layer3", "cp.resnet.layer4",
        "cp.arm16", "cp.arm32", "cp.conv_avg", "cp.conv_head16", "cp.conv_head32",
        "ffm", "conv_out.conv", "conv_out.conv_out",
    ]
    for name in tap_names:
        mod = dict(model.named_modules())[name]
        def mk(n):
            def hook(_m, _i, o):
                taps[n] = o.detach()
            return hook
        hooks.append(mod.register_forward_hook(mk(name)))

    bisenet_input = normalize(input_t.clone(), IMAGENET_MEAN, IMAGENET_STD)
    with torch.no_grad():
        out, out16, out32 = model(bisenet_input)
    for h in hooks:
        h.remove()

    save(
        out_dir,
        "bisenet.safetensors",
        {
            "input_normalized": bisenet_input[0],
            "out": out[0],
            **{f"tap.{k}": v[0] for k, v in taps.items()},
        },
        manifest,
    )

    # --- masking + resize, verbatim from pipeline_flux.py ---
    parsing_out = out.argmax(dim=1, keepdim=True)
    bg = sum(parsing_out == i for i in BG_LABELS).bool()
    white_image = torch.ones_like(input_t)
    face_features_image = torch.where(bg, white_image, to_gray(input_t))
    resized = resize(face_features_image, EVA_L336_SIZE, interpolation=InterpolationMode.BICUBIC)
    normalized = normalize(resized.clone(), EVA_MEAN, EVA_STD)

    save(
        out_dir,
        "masked.safetensors",
        {
            "parsing_class_map": parsing_out[0].to(torch.float32),
            "bg_mask": bg[0].to(torch.float32),
            "masked_512": face_features_image[0],
            "resized_336": resized[0],
            "eva_input_normalized_336": normalized[0],
        },
        manifest,
    )

    # `source` is the checkpoint-provenance block every dumper writes; the input
    # photograph, which used to hold that key, is `photo` - they are different
    # facts and only one of them decides whether these tensors can be compared
    # against a given set of weights.
    #
    # BiSeNet ships no config file, so `identity` is read off the state dict
    # that was just loaded: the segmentation class count and the width of the
    # ResNet-18 trunk's last stage are exactly what a differently-shaped
    # parsing checkpoint would disagree on, and `crates/bisenet::config` pins
    # both on the reading side.
    import facexlib

    weights = os.path.join(os.path.dirname(facexlib.__file__), "weights", "parsing_bisenet.pth")
    sd = model.state_dict()
    with open(os.path.join(out_dir, "face_parsing_manifest.json"), "w") as f:
        json.dump({
            "photo": os.path.abspath(args.photo),
            "kps": kps.tolist(),
            "files": manifest,
            "source": source_block(
                checkpoint="facexlib/parsing_bisenet",
                files=[weights] if os.path.exists(weights) else [],
                identity={
                    "num_class": int(sd["conv_out.conv_out.weight"].shape[0]),
                    "resnet_stage4_channels": int(sd["cp.resnet.layer4.1.conv2.weight"].shape[0]),
                },
            ),
        }, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
