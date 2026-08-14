#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump insightface / antelopev2 SCRFD reference goldens for brain's `crates/scrfd`.

The reference here is the **ONNX model itself** (`scrfd_10g_bnkps.onnx`, the
SCRFD-10GF detector), driven exactly the way `insightface.model_zoo.scrfd`
drives it. There is no PyTorch reference to hook, so per-stage activations are
captured by **promoting internal ONNX values to graph outputs** and re-running
the session - the ONNX equivalent of a forward hook, and exact rather than a
re-derivation.

Files written under `--out` (default `testdata/face/antelopev2`):

  scrfd.safetensors          synthetic 640x640 -> stem / C2..C5 / FPN / neck /
                             per-stride head features + the 9 raw head outputs +
                             decoded anchor centres
  manifest_scrfd.json        tensor shapes + sha256, tap->ONNX-value map, the
                             exact reference config, and library versions

`e2e.safetensors` - the real-photo golden `crates/scrfd`'s decode/NMS test also
replays - is NOT written here: an end-to-end run is detect THEN embed, so it
needs both released graphs and belongs to the pipeline dumper,
`arcface_dump_reference.py --photos`.

Everything is saved as f32 (brain's safetensors reader is F32/F16/BF16-only);
uint8 images are exactly representable. Fixed seed, CPU only.

Usage:
  python tools/goldens/scrfd_dump_reference.py \
      --weights /path/to/antelopev2 --out testdata/face/antelopev2 [--seed 1234]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper
from safetensors.torch import save_file
import cv2

# ---------------------------------------------------------------------------
# insightface reference constants (insightface/model_zoo/scrfd.py). Copied
# verbatim so this script does not need the (unbuildable-here) insightface
# package installed. NOTE the detector's std is 128.0 - the ArcFace embedder
# divides by 127.5, and the two are not interchangeable.
# ---------------------------------------------------------------------------
DET_INPUT_MEAN, DET_INPUT_STD = 127.5, 128.0     # scrfd.py
DET_THRESH, NMS_THRESH = 0.5, 0.4                # scrfd.py defaults
FEAT_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2                                  # 9 outputs -> fmc=3, na=2, kps


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def anchor_centers(height, width, stride, num_anchors):
    ac = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    ac = (ac * stride).reshape((-1, 2))
    if num_anchors > 1:
        ac = np.stack([ac] * num_anchors, axis=1).reshape((-1, 2))
    return ac


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def save(out, name, tensors, manifest, extra=None):
    t = {}
    for k, v in tensors.items():
        a = v.detach().cpu() if isinstance(v, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(v))
        t[k] = a.to(torch.float32).contiguous().clone()
    path = os.path.join(out, name)
    save_file(t, path)
    manifest[name] = {
        "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "tensors": {k: list(v.shape) for k, v in t.items()},
        "dtype": "F32 (all tensors; uint8 pixel values are exact)",
    }
    if extra:
        manifest[name].update(extra)
    print(f"wrote {name}: {len(t)} tensors, "
          f"{os.path.getsize(path) / 1e6:.1f} MB", flush=True)


def session_with_taps(model_path, taps, scratch):
    """Re-serialize `model_path` with `taps` (internal value names) promoted to
    graph outputs - the ONNX analogue of a forward hook - and open a session."""
    model = onnx.load(model_path)
    existing = {o.name for o in model.graph.output}
    produced = {o for n in model.graph.node for o in n.output}
    for name in taps:
        if name in existing:
            continue
        if name not in produced:
            raise SystemExit(f"tap {name!r} is not produced by any node in {model_path}")
        model.graph.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    os.makedirs(scratch, exist_ok=True)
    tapped = os.path.join(scratch, os.path.basename(model_path).replace(".onnx", "_tapped.onnx"))
    onnx.save(model, tapped, save_as_external_data=False)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # keep taps alive
    sess = ort.InferenceSession(tapped, so, providers=["CPUExecutionProvider"])
    return sess, [o.name for o in sess.get_outputs()]


def synth_image(h, w, seed, kind):
    """Deterministic uint8 BGR image: smooth analytic structure + seeded noise."""
    rng = np.random.default_rng(seed)
    ys = np.linspace(0.0, np.pi, h, dtype=np.float64)[:, None]
    xs = np.linspace(0.0, 2.0 * np.pi, w, dtype=np.float64)[None, :]
    if kind == "face":                                   # coarse face-ish blobs
        img = np.zeros((h, w, 3), dtype=np.float64)
        img[..., 0] = 0.5 + 0.35 * np.sin(3 * xs + ys)
        img[..., 1] = 0.5 + 0.30 * np.cos(2 * xs) * np.sin(1.5 * ys)
        img[..., 2] = 0.5 + 0.25 * (ys / np.pi * 2 - 1)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        for (cx, cy, r, amp) in [(0.34 * w, 0.46 * h, 0.09 * w, -0.35),
                                 (0.66 * w, 0.46 * h, 0.09 * w, -0.35),
                                 (0.50 * w, 0.64 * h, 0.10 * w, 0.20),
                                 (0.50 * w, 0.82 * h, 0.14 * w, -0.20)]:
            g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
            img += (amp * g)[..., None]
    else:
        img = np.stack([0.5 + 0.4 * np.sin(xs + ys + k) for k in (0.0, 2.1, 4.2)], -1)
    img = img + rng.normal(0.0, 0.02, img.shape)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def det_blob(bgr_u8):
    """insightface SCRFD.forward preprocessing (cv2.dnn.blobFromImage)."""
    size = (bgr_u8.shape[1], bgr_u8.shape[0])
    return cv2.dnn.blobFromImage(bgr_u8, 1.0 / DET_INPUT_STD, size,
                                 (DET_INPUT_MEAN,) * 3, swapRB=True)


# ---------------------------------------------------------------------------
# SCRFD
# ---------------------------------------------------------------------------
SCRFD_TAPS = {
    "stem_pre_pool": ("285", "Relu"), "stem": ("286", "MaxPool"),
    "c2": ("307", "Relu"), "c3": ("338", "Relu"),
    "c4": ("355", "Relu"), "c5": ("379", "Relu"),
    "lat3": ("380", "Conv"), "lat4": ("381", "Conv"), "lat5": ("382", "Conv"),
    "fpn4": ("402", "Add"), "fpn3": ("422", "Add"),
    "pafpn16_pre": ("424", "Conv"), "pafpn32_pre": ("425", "Conv"),
    "pafpn16": ("427", "Add"), "pafpn32": ("429", "Add"),
    "neck8": ("423", "Conv"), "neck16": ("430", "Conv"), "neck32": ("431", "Conv"),
    "head8_feat": ("440", "Relu"), "head16_feat": ("463", "Relu"),
    "head32_feat": ("486", "Relu"),
    "head8_cls_raw": ("441", "Conv"), "head16_cls_raw": ("464", "Conv"),
    "head32_cls_raw": ("487", "Conv"),
    "head8_bbox_scaled": ("443", "Mul"), "head16_bbox_scaled": ("466", "Mul"),
    "head32_bbox_scaled": ("489", "Mul"),
    "head8_kps_raw": ("444", "Conv"), "head16_kps_raw": ("467", "Conv"),
    "head32_kps_raw": ("490", "Conv"),
}


def check_scrfd_graph(path):
    g = onnx.load(path, load_external_data=False).graph
    producer = {o: n for n in g.node for o in n.output}
    for tap, (val, op) in SCRFD_TAPS.items():
        assert val in producer, f"scrfd tap {tap} ({val}) missing"
        assert producer[val].op_type == op, \
            f"scrfd tap {tap} ({val}) is {producer[val].op_type}, expected {op}"
    scales = {}
    from onnx import numpy_helper
    init = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type == "Mul":
            for i in n.input:
                if i in init:
                    scales[n.output[0]] = float(numpy_helper.to_array(init[i]))
    assert len(g.output) == 9, f"expected 9 outputs, got {len(g.output)}"
    return [o.name for o in g.output], scales


def scrfd_decode(net_outs, out_names, size):
    """insightface SCRFD.forward decode, threshold 0 (keep every anchor).

    Thresholding and greedy NMS are deliberately NOT here: on the synthetic
    image no anchor clears `det_thresh`, so they could not be gated from this
    file. brain's `nms` is gated against the real-photo detections in
    `e2e.safetensors` instead (see the module docstring)."""
    fmc, w, h = 3, size, size
    dec = {}
    for idx, stride in enumerate(FEAT_STRIDES):
        scores = net_outs[out_names[idx]]
        bbox = net_outs[out_names[idx + fmc]] * stride
        kps = net_outs[out_names[idx + 2 * fmc]] * stride
        ac = anchor_centers(h // stride, w // stride, stride, NUM_ANCHORS)
        dec[f"anchors_{stride}"] = ac
        dec[f"boxes_{stride}"] = distance2bbox(ac, bbox)
        dec[f"kps_{stride}"] = distance2kps(ac, kps).reshape(-1, 5, 2)
        dec[f"scores_{stride}"] = scores
    return dec


def dump_scrfd(args, manifest, scratch):
    path = os.path.join(args.weights, "scrfd_10g_bnkps.onnx")
    out_names, mul_scales = check_scrfd_graph(path)
    taps = {k: v[0] for k, v in SCRFD_TAPS.items()}
    sess, all_names = session_with_taps(path, list(taps.values()), scratch)
    in_name = sess.get_inputs()[0].name

    img = synth_image(640, 640, args.seed + 1, "face")
    blob = det_blob(img)
    res = dict(zip(all_names, sess.run(all_names, {in_name: blob})))
    dec = scrfd_decode(res, out_names, 640)

    # self-check: promoting taps to outputs (and disabling ORT graph fusion)
    # must not move the 9 head tensors
    plain = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = plain.run(None, {plain.get_inputs()[0].name: blob})
    d = max(float(np.abs(res[n] - r).max()) for n, r in zip(out_names, ref))
    assert d < 1e-5, f"tapped scrfd session diverges from the plain one: {d:.3e}"
    print(f"  tapped vs plain head outputs max abs diff {d:.3e}", flush=True)
    dp = float(np.abs(((img[:, :, ::-1].astype(np.float32) - DET_INPUT_MEAN)
                       / DET_INPUT_STD).transpose(2, 0, 1)[None] - blob).max())
    assert dp < 1e-5, f"blobFromImage != manual (BGR->RGB, -mean)/std: {dp:.3e}"

    t = {"image_bgr_u8": img.astype(np.float32), "blob": blob[0]}
    for k, v in taps.items():
        t[k] = res[v][0]
    for i, n in enumerate(out_names):
        stride = FEAT_STRIDES[i % 3]
        kind = ["score", "bbox", "kps"][i // 3]
        t[f"out_{kind}_{stride}"] = res[n]
    # scores_* are byte-identical to out_score_* (the decode does not touch them)
    t.update({k: v for k, v in dec.items() if not k.startswith("scores_")})
    npos = {s: int((dec[f"scores_{s}"] >= DET_THRESH).sum()) for s in FEAT_STRIDES}
    save(args.out, "scrfd.safetensors", t, manifest, {
        "onnx_value_of_tap": taps,
        "graph_output_order": out_names,
        "graph_output_meaning": "scores[8,16,32], bbox[8,16,32], kps[8,16,32]",
        "bbox_mul_scale_per_output": mul_scales,
        "preprocess": "cv2.dnn.blobFromImage(bgr_u8, 1/128.0, (640,640), "
                      "(127.5,)*3, swapRB=True) -> NCHW RGB",
        "decode": "bbox/kps *= stride; anchors = mgrid[:H,:W][::-1]*stride "
                  "repeated num_anchors=2; boxes=distance2bbox, kps=distance2kps",
        "det_thresh": DET_THRESH, "nms_thresh": NMS_THRESH,
        "synthetic_positives_per_stride": npos,
        "note": "the synthetic image is not a real face; positives above "
                "det_thresh may be zero. The 9 raw head tensors are the golden.",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="directory holding scrfd_10g_bnkps.onnx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scratch", default=None,
                    help="where the tapped ONNX copy goes (default: a temp dir, "
                         "removed on exit; it is regenerated on every run)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="scrfd_taps_") if args.scratch is None else None
    scratch = args.scratch or tmp
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = {}
    print("== SCRFD / scrfd_10g_bnkps", flush=True)
    dump_scrfd(args, manifest, scratch)

    manifest["config"] = {
        "seed": args.seed,
        "weights_dir": os.path.abspath(args.weights),
        "model": "scrfd_10g_bnkps.onnx (SCRFD-10GF, 9 outputs -> fmc 3, "
                 "strides [8,16,32], num_anchors 2, use_kps, opset 11, ir 6)",
        "scrfd": {"input_mean": DET_INPUT_MEAN, "input_std": DET_INPUT_STD,
                  "input_size": [640, 640], "layout": "NCHW RGB",
                  "det_thresh": DET_THRESH, "nms_thresh": NMS_THRESH,
                  "strides": FEAT_STRIDES, "num_anchors": NUM_ANCHORS},
        "versions": {"numpy": np.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__, "opencv": cv2.__version__,
                     "torch": torch.__version__,
                     "insightface_reference": "1.0.1 (source, vendored inline)"},
    }
    with open(os.path.join(args.out, "manifest_scrfd.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"done -> {args.out}/manifest_scrfd.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
