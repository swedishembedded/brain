#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump OpenSeeFace reference goldens for brain's `crates/openseeface`.

The reference is the released ONNX graphs themselves, run by onnxruntime on
the CPU, exactly as OpenSeeFace's `tracker.py` drives them. Per-stage
activations are captured by promoting internal values to graph outputs - the
ONNX equivalent of a forward hook - so a stage tap is the value the shipped
graph computed, not a re-derivation.

What is dumped, per model (`detector`, `lm0`, `lm3`):

  <model>.safetensors     two inputs, each with every stage tap:
                            syn.*   a seeded N(0,1) input in normalized space
                            real.*  the portrait, preprocessed as tracker.py does
                          tap names: `stem`, `blocks.<s>.<b>`, `up<k>`, `out`
                          (+ `maxpool` for the detector)
  pipeline.safetensors    the real-photo pipeline: the frame, the detector's
                          box, the landmark crop and its input tensor, and the
                          decoded 66 landmarks (y, x, confidence) of lm0 and lm3
  frame.png               the frame the pipeline ran on (the portrait,
                          downscaled once with INTER_AREA so it is a webcam-
                          sized input)
  manifest.json           shapes + sha256, tap -> ONNX value, the source block

The decode below follows OpenSeeFace `tracker.py` (BSD-2-Clause, Copyright (c)
2019 Emiliana) at commit 50c09741: `Tracker.detect_faces`, the crop in
`Tracker.predict`, `clamp_to_im`, `Tracker.preprocess`, `Tracker.landmarks`
and `logit_arr`, reproduced here so the goldens record what the reference
tracker would report. See NOTICE.md.

Usage:
  python tools/goldens/openseeface_dump_reference.py \\
      --models DIR_WITH_ONNX --photo portrait.jpg --out testdata/face/openseeface
"""

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
from safetensors.numpy import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402

FILES = {"detector": "mnv3_detection_opt.onnx", "lm0": "lm_model0_opt.onnx", "lm3": "lm_model3_opt.onnx"}
MEAN = np.float32([0.485, 0.456, 0.406])
STD = np.float32([0.229, 0.224, 0.225])
# tracker.py folds ImageNet normalisation into one multiply-add on 0..255 RGB
SCALE = 1.0 / (STD * 255.0)
OFFSET = -MEAN / STD
FRAME_WIDTH = 640


def group_of(weight_name):
    """The block a conv weight belongs to: `stem`, `blocks.s.b`, `up<k>`, `group`."""
    n = weight_name
    if n.startswith("ConvBnFusion_W_"):
        n = n[len("ConvBnFusion_W_"):].rsplit("_", 1)[0]
    if n.startswith("conv_stem"):
        return "stem"
    if n.startswith("blocks."):
        return ".".join(n.split(".")[:3])
    return n.split(".")[0]


SKIP_BACK = {"Shape", "Slice", "Concat", "Resize", "Constant", "Gather", "Unsqueeze", "Cast"}


def taps(graph):
    """tap name -> ONNX value: each block's output is the last node before
    the next block's first conv, stepping back over the decoder's
    shape-plumbing nodes."""
    nodes = list(graph.node)
    firsts = []
    for i, n in enumerate(nodes):
        if n.op_type in ("Conv", "FusedConv"):
            g = group_of(n.input[1])
            if not firsts or firsts[-1][0] != g:
                firsts.append((g, i))
    out = {}
    for k, (g, i) in enumerate(firsts):
        if g == "group":
            continue
        end = firsts[k + 1][1] - 1 if k + 1 < len(firsts) else len(nodes) - 1
        while nodes[end].op_type in SKIP_BACK:
            end -= 1
        out[g] = nodes[end].output[0]
    return out


def session(path, extra):
    m = onnx.load(path)
    have = {o.name for o in m.graph.output}
    for v in extra:
        if v not in have:
            m.graph.output.append(helper.make_tensor_value_info(v, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"]), m


def run(path, x):
    m0 = onnx.load(path)
    tp = taps(m0.graph)
    sess, m = session(path, list(tp.values()))
    names = [o.name for o in m.graph.output]
    res = dict(zip(names, sess.run(names, {"input": x})))
    out = {"out": res["output"]}
    if "maxpool" in res:
        out["maxpool"] = res["maxpool"]
    for t, v in tp.items():
        out[t] = res[v]
    return out, tp


def normalize(rgb):
    return np.ascontiguousarray((rgb.astype(np.float32) * SCALE + OFFSET).transpose(2, 0, 1)[None], dtype=np.float32)


def saved(tensors):
    """safetensors' numpy writer copies an array's raw buffer, so a strided
    view (a transpose, a channel flip) would be written scrambled. Every
    saved tensor goes through here."""
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()}


def detect(det_out, frame_w, frame_h, threshold=0.6):
    """tracker.py `detect_faces` for one face: the strongest peak that
    survives the 3x3 max-pool, as (x, y, w, h) in frame pixels."""
    hm, mp = det_out["out"][0, 0].copy(), det_out["maxpool"][0, 0]
    hm[hm != mp] = 0
    i = int(np.argmax(hm))
    y, x = i // 56, i % 56
    c = hm[y, x]
    r = det_out["out"][0, 1, y, x] * 112.0
    if c < threshold:
        raise SystemExit(f"no face above {threshold} (best {c:.3f}); pick another photo")
    box = np.float32([x * 4 - r, y * 4 - r, 2 * r, 2 * r])
    box[[0, 2]] *= frame_w / 224.0
    box[[1, 3]] *= frame_h / 224.0
    return box, float(c)


def clamp_to_im(pt, w, h):
    x, y = pt
    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)
    return int(x), int(y + 1)


def crop_of(box, w, h):
    x, y, bw, bh = [float(v) for v in box]
    x1, y1 = clamp_to_im((x - int(bw * 0.1), y - int(bh * 0.125)), w, h)
    x2, y2 = clamp_to_im((x + bw + int(bw * 0.1), y + bh + int(bh * 0.125)), w, h)
    return x1, y1, x2, y2


def landmarks(t, crop, res=224.0, out_res=27.0, logit_factor=16.0):
    """tracker.py `landmarks`: per landmark, the heatmap argmax plus its
    logit-decoded sub-cell offset, mapped back through the crop."""
    x1, y1, x2, y2 = crop
    sx, sy = (x2 - x1) / res, (y2 - y1) / res
    r = res - 1
    n = int(out_res) + 1
    main = t[0:66].reshape(66, n * n)
    m = main.argmax(1)
    conf = np.take_along_axis(main, m[:, None], 1)[:, 0]
    ox = np.take_along_axis(t[66:132].reshape(66, n * n), m[:, None], 1)[:, 0]
    oy = np.take_along_axis(t[132:198].reshape(66, n * n), m[:, None], 1)[:, 0]
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))) / logit_factor
    ox, oy = r * lg(ox), r * lg(oy)
    ty = y1 + sy * (r * np.floor(m / n) / out_res + ox)
    tx = x1 + sx * (r * np.floor(np.mod(m, n)) / out_res + oy)
    return np.stack([ty, tx, conf], 1).astype(np.float32)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--out", default="testdata/face/openseeface")
    ap.add_argument("--seed", type=int, default=None, help="seed of the synthetic input (default: drawn, printed and recorded)")
    a = ap.parse_args()
    if a.seed is None:
        a.seed = int.from_bytes(os.urandom(4), "little")
    print(f"seed {a.seed}")
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.RandomState(a.seed)

    photo = cv2.imread(a.photo, cv2.IMREAD_COLOR)
    if photo is None:
        raise SystemExit(f"cannot read {a.photo}")
    fh = int(round(photo.shape[0] * FRAME_WIDTH / photo.shape[1]))
    frame = cv2.resize(photo, (FRAME_WIDTH, fh), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(a.out, "frame.png"), frame)
    frame = cv2.imread(os.path.join(a.out, "frame.png"), cv2.IMREAD_COLOR)  # exactly what the test reads
    h, w = frame.shape[:2]

    manifest = {"seed": a.seed, "files": {}, "taps": {}, "versions": {"onnxruntime": ort.__version__, "onnx": onnx.__version__, "opencv": cv2.__version__, "numpy": np.__version__}}
    pipeline = {"frame": frame[:, :, ::-1].astype(np.float32)}  # RGB, 0..255

    # detector: tracker.py resizes the uint8 BGR frame, flips to RGB, normalizes
    det_in = normalize(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)[:, :, ::-1])
    syn = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    path = os.path.join(a.models, FILES["detector"])
    t_syn, tp = run(path, syn)
    t_real, _ = run(path, det_in)
    box, score = detect(t_real, w, h)
    crop = crop_of(box, w, h)
    pipeline.update({"det.input": det_in, "det.box": box, "det.score": np.float32([score]), "lm.crop": np.float32(crop)})
    dumps = {"detector": (t_syn, t_real, tp, syn, det_in)}

    x1, y1, x2, y2 = crop
    patch = cv2.resize(np.float32(frame[y1:y2, x1:x2, ::-1]), (224, 224), interpolation=cv2.INTER_LINEAR)
    lm_in = normalize(patch)
    pipeline["lm.input"] = lm_in
    for key in ("lm0", "lm3"):
        path = os.path.join(a.models, FILES[key])
        t_syn, tp = run(path, syn)
        t_real, _ = run(path, lm_in)
        pipeline[f"{key}.landmarks"] = landmarks(t_real["out"][0], crop)
        dumps[key] = (t_syn, t_real, tp, syn, lm_in)

    for key, (t_syn, t_real, tp, x_syn, x_real) in dumps.items():
        tensors = {"syn.input": x_syn, "real.input": x_real}
        for k, v in t_syn.items():
            tensors[f"syn.{k}"] = v.astype(np.float32)
        for k, v in t_real.items():
            tensors[f"real.{k}"] = v.astype(np.float32)
        save_file(saved(tensors), os.path.join(a.out, f"{key}.safetensors"))
        manifest["taps"][key] = tp
        manifest["files"][f"{key}.safetensors"] = {k: {"shape": list(v.shape), "sha256": sha(v)} for k, v in tensors.items()}
    save_file(saved(pipeline), os.path.join(a.out, "pipeline.safetensors"))
    manifest["files"]["pipeline.safetensors"] = {k: {"shape": list(v.shape), "sha256": sha(v)} for k, v in pipeline.items()}
    manifest["pipeline"] = {"box_xywh": [float(v) for v in box], "score": score, "crop_x1y1x2y2": list(crop), "frame_wh": [w, h]}
    manifest["source"] = source_block(
        # the stem widths tell the three graphs apart; input and landmark count
        # fix every other dumped shape
        identity={"input": 224, "landmarks": 66, "detector_stem": 8, "lm0_stem": 8, "lm3_stem": 16},
        checkpoint="emilianavt/OpenSeeFace@50c09741897010faad721a218b2bbe14ecad6251",
        files=[os.path.join(a.models, f) for f in FILES.values()],
    )
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"box {box} score {score:.3f} crop {crop}")
    for key in ("lm0", "lm3"):
        lm = pipeline[f"{key}.landmarks"]
        print(f"{key}: mean confidence {lm[:, 2].mean():.3f}, eye corners (y, x) {lm[36, :2]} {lm[45, :2]}")


if __name__ == "__main__":
    main()
