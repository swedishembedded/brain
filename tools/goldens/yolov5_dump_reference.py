#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for proving that an edge detector
# matches its trained reference, number for number, for its clients. If your team
# needs expertise in numerical parity gates for accelerator ports then you can
# procure our services by sending an email to info@swedishembedded.com.

"""Dump onnxruntime goldens for `crates/yolov5`'s parity tests.

Runs the ONNX export (`yolov5s_relu.onnx`) under onnxruntime (CPU, fp32) at
320x320 and 640x640 and writes, under `<testdata>/golden/yolov5/` (gitignored,
never committed):

  golden.safetensors   every tensor below, f32
  manifest.json        shapes, sha256, run parameters, library versions and the
                       `source` block `brain_testutil::golden::Source` enforces

Tensors, for each size S in {320, 640}:

  noise{S}             [3,S,S]   deterministic uniform noise input (seeded)
  noise{S}_out{i}      [255,S/8>>i,S/8>>i] the three head outputs for it, i = 0..2
                       (post-sigmoid, exactly what the ONNX graph returns)
  img{S}               [3,S,S]   the real image, letterboxed (see below)
  img{S}_out{i}        the three head outputs for it
  img{S}_dets          [n,6]     detections after decode + NMS, original-image
                       pixels: x1,y1,x2,y2,conf,class
  image_rgb            [h0,w0,3] the decoded real image as 0..255 floats (so the
                       Rust test never depends on a JPEG decoder agreeing)

Decode and NMS are written here independently of the Rust implementation, from
the published YOLOv5 v6 formulas (xy = (2s - 0.5 + grid) * stride, wh = (2s)^2 *
anchor, score = objectness * class, class-aware greedy NMS with strict `>`).
The letterbox mirrors `imaging::letterbox_rgb` operation for operation in f32
(half-pixel nearest resize, 114/255 grey pad) so both sides feed the network
the same pixels.

The ONNX is exported with a fixed 640 input; its Resize nodes carry constant
scales, so the graph is shape-agnostic, and the input/output dims are made
dynamic in memory (the file is not modified) to run 320.

Usage:
  python tools/goldens/yolov5_dump_reference.py --onnx yolov5s_relu.onnx \\
      --image photo.jpg [--testdata testdata]

Requires: numpy, onnx, onnxruntime, pillow, safetensors.
"""

import argparse
import json
import os
import secrets
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import sha256_of, source_block  # noqa: E402

SIZES = (320, 640)
NC = 80
STRIDES = (8, 16, 32)
ANCHORS = (
    (10, 13, 16, 30, 33, 23),
    (30, 61, 62, 45, 59, 119),
    (116, 90, 156, 198, 373, 326),
)
CONF, IOU, MAX_DET = 0.25, 0.45, 300
PAD = np.float32(114.0 / 255.0)


def session_with_dynamic_hw(path):
    model = onnx.load(path)
    for d, name in zip(model.graph.input[0].type.tensor_type.shape.dim[2:], ("h", "w")):
        d.ClearField("dim_value")
        d.dim_param = name
    for out in model.graph.output:
        del out.type.tensor_type.shape.dim[:]
    del model.graph.value_info[:]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # deterministic summation order
    return ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])


def run(sess, chw):
    outs = sess.run(None, {"images": chw[None].astype(np.float32)})
    return [o[0].astype(np.float32) for o in outs]


def round_half_away(x):
    return np.floor(x + np.float32(0.5))  # inputs here are non-negative


def letterbox(rgb_u8, size):
    """`imaging::letterbox_rgb` in numpy, f32 throughout. Returns CHW and the transform."""
    h0, w0, _ = rgb_u8.shape
    src = rgb_u8.astype(np.float32) / np.float32(255.0)
    s = min(np.float32(size) / np.float32(w0), np.float32(size) / np.float32(h0))
    new_w = int(round_half_away(np.float32(w0) * s))
    new_h = int(round_half_away(np.float32(h0) * s))
    pad_x = (np.float32(size) - np.float32(new_w)) * np.float32(0.5)
    pad_y = (np.float32(size) - np.float32(new_h)) * np.float32(0.5)
    inv = np.float32(1.0) / s
    out = np.full((3, size, size), PAD, dtype=np.float32)
    ys = np.clip(round_half_away((np.arange(new_h, dtype=np.float32) + np.float32(0.5)) * inv - np.float32(0.5)), 0, h0 - 1).astype(int)
    xs = np.clip(round_half_away((np.arange(new_w, dtype=np.float32) + np.float32(0.5)) * inv - np.float32(0.5)), 0, w0 - 1).astype(int)
    dy, dx = int(pad_y), int(pad_x)  # truncation, as in the Rust
    out[:, dy:dy + new_h, dx:dx + new_w] = src[ys][:, xs].transpose(2, 0, 1)
    return out, (float(s), float(pad_x), float(pad_y))


def decode(heads, conf=CONF):
    """Candidates [x1,y1,x2,y2,conf,cls] in input pixels from the three head maps."""
    cands = []
    for i, head in enumerate(heads):
        c, h, w = head.shape
        p = head.reshape(3, 5 + NC, h, w)
        gy, gx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        for a in range(3):
            obj = p[a, 4]
            scores = obj[None] * p[a, 5:]
            best = scores.max(0)
            cls = scores.argmax(0)
            keep = (obj > conf) & (best > conf)
            cx = (p[a, 0] * 2 - 0.5 + gx) * STRIDES[i]
            cy = (p[a, 1] * 2 - 0.5 + gy) * STRIDES[i]
            bw = (p[a, 2] * 2) ** 2 * ANCHORS[i][2 * a]
            bh = (p[a, 3] * 2) ** 2 * ANCHORS[i][2 * a + 1]
            for y, x in zip(*np.nonzero(keep)):
                cands.append([cx[y, x] - bw[y, x] / 2, cy[y, x] - bh[y, x] / 2, cx[y, x] + bw[y, x] / 2, cy[y, x] + bh[y, x] / 2, best[y, x], cls[y, x]])
    return np.array(cands, dtype=np.float32).reshape(-1, 6)


def iou(a, b):
    iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
    ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    inter = iw * ih
    ua = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0) + max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0) - inter
    return inter / max(ua, 1e-9)


def nms(cands, thr=IOU, max_det=MAX_DET):
    order = sorted(range(len(cands)), key=lambda i: (-cands[i][4], i))
    keep = []
    for i in order:
        if len(keep) >= max_det:
            break
        if all(not (k[5] == cands[i][5] and iou(k, cands[i]) > thr) for k in keep):
            keep.append(cands[i])
    return np.array(keep, dtype=np.float32).reshape(-1, 6)


def unletterbox(dets, lb, w0, h0):
    s, px, py = lb
    out = dets.copy()
    inv = 1.0 / s
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - px) * inv, 0, w0)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - py) * inv, 0, h0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True, help="any photograph with detectable objects")
    ap.add_argument("--testdata", default="testdata")
    ap.add_argument("--seed", type=int, help="noise-input seed (default: drawn at random, printed and recorded in the manifest)")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else secrets.randbits(31)
    print(f"noise seed {seed}", file=sys.stderr)
    sess = session_with_dynamic_hw(args.onnx)
    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    h0, w0, _ = rgb.shape

    tensors, notes = {}, {}
    tensors["image_rgb"] = rgb.astype(np.float32)
    rng = np.random.RandomState(seed)
    for size in SIZES:
        noise = rng.uniform(0.0, 1.0, size=(3, size, size)).astype(np.float32)
        tensors[f"noise{size}"] = noise
        for i, o in enumerate(run(sess, noise)):
            tensors[f"noise{size}_out{i}"] = o

        chw, lb = letterbox(rgb, size)
        tensors[f"img{size}"] = chw
        heads = run(sess, chw)
        for i, o in enumerate(heads):
            tensors[f"img{size}_out{i}"] = o
        cands = decode(heads)
        dets = unletterbox(nms(cands), lb, w0, h0)
        tensors[f"img{size}_dets"] = dets
        # How close the nearest threshold decision is: a parity test is only
        # meaningful if no candidate sits within fp32 noise of the cut.
        margin = float(np.min(np.abs(cands[:, 4] - CONF))) if len(cands) else 1.0
        notes[f"img{size}"] = {"candidates": int(len(cands)), "detections": int(len(dets)), "min_conf_margin": margin}
        print(f"{size}: {len(cands)} candidates, {len(dets)} detections, min |conf - {CONF}| = {margin:.4g}", file=sys.stderr)

    out_dir = os.path.join(args.testdata, "golden", "yolov5")
    os.makedirs(out_dir, exist_ok=True)
    blob = os.path.join(out_dir, "golden.safetensors")
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()}, blob)

    params = sum(int(numpy_helper.to_array(t).size) for t in onnx.load(args.onnx).graph.initializer)
    manifest = {
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
        "sha256": {"golden.safetensors": sha256_of(blob)},
        "run": {"sizes": list(SIZES), "seed": seed, "conf": CONF, "iou": IOU, "max_det": MAX_DET, "image_size": [w0, h0], "ort_threads": 1},
        "notes": notes,
        "versions": {"onnxruntime": ort.__version__, "onnx": onnx.__version__, "numpy": np.__version__},
        "source": source_block(
            checkpoint="yolov5s_relu.onnx",
            files=[args.onnx],
            identity={"nc": NC, "head_channels": 3 * (5 + NC), "param_count": params},
        ),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {blob}", file=sys.stderr)


if __name__ == "__main__":
    main()
