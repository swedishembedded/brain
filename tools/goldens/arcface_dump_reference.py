#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump insightface / antelopev2 reference goldens for brain's `crates/arcface`.

The reference here is the **ONNX model itself** (`glintr100.onnx` = ArcFace
IResNet-100), driven exactly the way `insightface.model_zoo.arcface_onnx` drives
it, plus `insightface.utils.face_align` for the 5-point similarity-transform
alignment. There is no PyTorch reference to hook, so per-stage activations are
captured by **promoting internal ONNX values to graph outputs** and re-running
the session - the ONNX equivalent of a forward hook, and exact rather than a
re-derivation.

This is also the PIPELINE dumper: `--photos` runs the real end-to-end chain,
which is detect (`scrfd_10g_bnkps.onnx`, run as shipped) then align then embed,
so it needs both released graphs. The detector's own per-stage goldens are the
sibling script's, `scrfd_dump_reference.py`.

Files written under `--out` (default `testdata/face/antelopev2`):

  arcface.safetensors        synthetic 112x112 face -> preprocessing blob ->
                             stem / layer1..layer4 / bn2 / flatten / fc /
                             embedding taps, plus first-block internals per stage
  arcface_blocks.safetensors every residual block output (49) for bisection
  align.safetensors          arcface 5-point template, synthetic landmark sets,
                             the similarity matrix M, the cv2.warpAffine result,
                             and the equivalent normalized grid_sample grid+result
  e2e.safetensors            (optional, --photos) real photo -> detect -> align ->
                             embed, with the cosine matrix between embeddings.
                             Replayed by BOTH crates' parity tests: the detector's
                             decode/NMS gate needs real detections, and the
                             synthetic detector image has none above threshold.
  manifest.json              per-file tensor shapes + sha256, tap->ONNX-value map,
                             the exact reference config, and library versions

Everything is saved as f32 (brain's safetensors reader is F32/F16/BF16-only);
uint8 images are exactly representable. Fixed seed, CPU only.

Usage:
  python tools/goldens/arcface_dump_reference.py \
      --weights /path/to/antelopev2 --out testdata/face/antelopev2 \
      [--seed 1234] [--photos a.jpg b.jpg ...]
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
from skimage import transform as sktrans
import cv2

# ---------------------------------------------------------------------------
# insightface reference constants (insightface/utils/face_align.py,
# insightface/model_zoo/{arcface_onnx,scrfd}.py). Copied verbatim so this script
# does not need the (unbuildable-here) insightface package installed.
# ---------------------------------------------------------------------------
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

ARC_INPUT_MEAN, ARC_INPUT_STD = 127.5, 127.5     # arcface_onnx.py
DET_INPUT_MEAN, DET_INPUT_STD = 127.5, 128.0     # scrfd.py  (NOTE: std differs)
DET_THRESH, NMS_THRESH = 0.5, 0.4                # scrfd.py defaults
FEAT_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2                                  # 9 outputs -> fmc=3, na=2, kps


def estimate_norm(lmk, image_size=112):
    """insightface.utils.face_align.estimate_norm (mode='arcface')."""
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio, diff_x = float(image_size) / 112.0, 0.0
    else:
        ratio, diff_x = float(image_size) / 128.0, 8.0 * float(image_size) / 128.0
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    if hasattr(sktrans.SimilarityTransform, "from_estimate"):  # skimage >= 0.26
        tform = sktrans.SimilarityTransform.from_estimate(lmk, dst)
    else:
        tform = sktrans.SimilarityTransform()
        tform.estimate(lmk, dst)
    return tform.params[0:2, :]


def umeyama_similarity(src, dst):
    """Independent least-squares similarity solve (Umeyama 1991), the check for
    `estimate_norm`'s skimage call. Deliberately does NOT share code with it."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    u, d, vt = np.linalg.svd(cov)
    s = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1] = -1
    r = u @ np.diag(s) @ vt
    var_s = (xs ** 2).sum() / n
    scale = (d * s).sum() / var_s
    t = mu_d - scale * (r @ mu_s)
    m = np.zeros((2, 3))
    m[:, :2] = scale * r
    m[:, 2] = t
    return m


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


def nms(dets, thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


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


def arcface_blob(bgr_u8):
    """insightface ArcFaceONNX.get_feat preprocessing (cv2.dnn.blobFromImages)."""
    return cv2.dnn.blobFromImages([bgr_u8], 1.0 / ARC_INPUT_STD, (112, 112),
                                  (ARC_INPUT_MEAN,) * 3, swapRB=True)


def det_blob(bgr_u8):
    """insightface SCRFD.forward preprocessing (cv2.dnn.blobFromImage)."""
    size = (bgr_u8.shape[1], bgr_u8.shape[0])
    return cv2.dnn.blobFromImage(bgr_u8, 1.0 / DET_INPUT_STD, size,
                                 (DET_INPUT_MEAN,) * 3, swapRB=True)


# ---------------------------------------------------------------------------
# ArcFace / IResNet-100
# ---------------------------------------------------------------------------
def arcface_taps(model_path):
    """Derive stage/block tap names from the graph instead of hardcoding them.

    Exported IResNet-100 shape: Conv+PReLU stem (BN folded into the conv), then
    49 residual blocks each ending in an `Add`, grouped into 4 stages by channel
    count [3, 13, 30, 3], then BN -> Flatten -> Gemm -> BN(features)."""
    g = onnx.load(model_path, load_external_data=False).graph
    init = {i.name: i for i in g.initializer}
    prelus = [n for n in g.node if n.op_type == "PRelu"]
    assert g.node[0].op_type == "Conv" and g.node[1].op_type == "PRelu", "unexpected stem"
    stem = g.node[1].output[0]

    blocks, ch = [], []
    last_conv_oc = None
    for n in g.node:
        if n.op_type == "Conv":
            w = [i for i in n.input if i in init][0]
            last_conv_oc = init[w].dims[0]
        elif n.op_type == "Add":
            blocks.append(n.output[0])
            ch.append(last_conv_oc)
    groups, cur = [], [0]
    for i in range(1, len(blocks)):
        if ch[i] != ch[i - 1]:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    counts = [len(x) for x in groups]
    assert counts == [3, 13, 30, 3], f"not IResNet-100: block counts {counts}"

    flat = [n for n in g.node if n.op_type == "Flatten"][0]
    gemm = [n for n in g.node if n.op_type == "Gemm"][0]
    bn2 = flat.input[0]
    stages = {f"layer{k + 1}": blocks[grp[-1]] for k, grp in enumerate(groups)}

    # first block of each stage, internals (conv1 / prelu / conv2 / shortcut)
    inner = {}
    for k, grp in enumerate(groups):
        b = blocks[grp[0]]
        add = [n for n in g.node if n.output[0] == b][0]
        conv2 = [n for n in g.node if n.output[0] == add.input[0]][0]
        prelu = [n for n in g.node if n.output[0] == conv2.input[0]][0]
        conv1 = [n for n in g.node if n.output[0] == prelu.input[0]][0]
        inner[f"s{k + 1}b0_conv1"] = conv1.output[0]
        inner[f"s{k + 1}b0_prelu"] = prelu.output[0]
        inner[f"s{k + 1}b0_conv2"] = conv2.output[0]
        inner[f"s{k + 1}b0_branch"] = add.input[1]
        inner[f"s{k + 1}b0_bn_in"] = conv1.input[0]
    named = {"stem": stem, **stages, "bn2": bn2, "flatten": flat.output[0],
             "fc": gemm.output[0], "embedding": g.output[0].name, **inner}
    per_block = {f"block{i:02d}": b for i, b in enumerate(blocks)}
    print(f"  IResNet: {len(prelus)} PReLU, {len(blocks)} residual blocks "
          f"{counts}, channels {[ch[grp[0]] for grp in groups]}", flush=True)
    return named, per_block


def dump_arcface(args, manifest, scratch):
    path = os.path.join(args.weights, "glintr100.onnx")
    named, per_block = arcface_taps(path)
    taps = {**named, **per_block}
    sess, out_names = session_with_taps(path, list(taps.values()), scratch)
    in_name = sess.get_inputs()[0].name

    aligned = synth_image(112, 112, args.seed, "face")           # BGR uint8
    blob = arcface_blob(aligned)
    res = dict(zip(out_names, sess.run(out_names, {in_name: blob})))
    emb = res[named["embedding"]].reshape(-1)

    # self-check: the promoted-output session must not change the embedding
    plain = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = plain.run(None, {plain.get_inputs()[0].name: blob})[0].reshape(-1)
    d = float(np.abs(emb - ref).max())
    assert d < 1e-5, f"tapped session diverges from the plain one: {d:.3e}"
    print(f"  tapped vs plain embedding max abs diff {d:.3e}", flush=True)
    # self-check: preprocessing reproduced from first principles
    manual = ((aligned[:, :, ::-1].astype(np.float32) - ARC_INPUT_MEAN) / ARC_INPUT_STD)
    manual = manual.transpose(2, 0, 1)[None]
    dp = float(np.abs(manual - blob).max())
    assert dp < 1e-5, f"blobFromImages != manual (BGR->RGB, -mean)/std: {dp:.3e}"

    t = {"aligned_bgr_u8": aligned.astype(np.float32),
         "blob": blob[0],
         "embedding_normed": emb / np.linalg.norm(emb)}
    for k, v in named.items():
        t[k] = res[v][0] if res[v].ndim == 4 else res[v].reshape(-1)
    save(args.out, "arcface.safetensors", t, manifest,
         {"onnx_value_of_tap": named,
          "embedding_l2_norm": float(np.linalg.norm(emb)),
          "preprocess": "cv2.dnn.blobFromImages(bgr_u8, 1/127.5, (112,112), "
                        "(127.5,)*3, swapRB=True) -> NCHW RGB"})
    save(args.out, "arcface_blocks.safetensors",
         {k: res[v][0] for k, v in per_block.items()}, manifest,
         {"onnx_value_of_tap": per_block,
          "note": "output of every residual Add, in graph order; "
                  "stages are blocks 0-2 / 3-15 / 16-45 / 46-48"})
    return emb


# ---------------------------------------------------------------------------
# SCRFD - run AS SHIPPED, only to drive the end-to-end chain. Its per-stage
# goldens (tapped internals, the 9 raw head outputs) are the sibling script's:
# tools/goldens/scrfd_dump_reference.py.
# ---------------------------------------------------------------------------
def det_session(weights):
    """Open the released detector and return `(session, input, 9 output names)`."""
    path = os.path.join(weights, "scrfd_10g_bnkps.onnx")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    assert len(out_names) == 9, f"expected 9 outputs, got {len(out_names)}"
    return sess, sess.get_inputs()[0].name, out_names


def scrfd_decode(net_outs, out_names, size):
    """insightface SCRFD.forward decode, threshold 0 (keep every anchor)."""
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


def detect(sess, in_name, out_names, bgr_u8, size=640):
    """insightface SCRFD.detect at a single input size (the classic path used by
    FaceAnalysis(det_size=(640, 640)))."""
    h0, w0 = bgr_u8.shape[:2]
    im_ratio = float(h0) / w0
    if im_ratio > 1.0:
        nh, nw = size, int(size / im_ratio)
    else:
        nw, nh = size, int(size * im_ratio)
    det_scale = float(nh) / h0
    det_img = np.zeros((size, size, 3), dtype=np.uint8)
    det_img[:nh, :nw, :] = cv2.resize(bgr_u8, (nw, nh))
    blob = det_blob(det_img)
    outs = dict(zip(out_names, sess.run(out_names, {in_name: blob})))
    dec = scrfd_decode(outs, out_names, size)

    scores, boxes, kpss = [], [], []
    for stride in FEAT_STRIDES:
        s = dec[f"scores_{stride}"].ravel()
        keep = np.where(s >= DET_THRESH)[0]
        scores.append(s[keep, None])
        boxes.append(dec[f"boxes_{stride}"][keep])
        kpss.append(dec[f"kps_{stride}"][keep])
    scores = np.vstack(scores)
    if scores.size == 0:
        return det_img, det_scale, blob, outs, dec, np.zeros((0, 5), np.float32), \
            np.zeros((0, 5, 2), np.float32)
    order = scores.ravel().argsort()[::-1]
    pre = np.hstack([np.vstack(boxes) / det_scale, scores]).astype(np.float32)[order]
    kps = (np.vstack(kpss) / det_scale)[order]
    keep = nms(pre, NMS_THRESH)
    return det_img, det_scale, blob, outs, dec, pre[keep], kps[keep]


# ---------------------------------------------------------------------------
# alignment: similarity transform + warp
# ---------------------------------------------------------------------------
def warp_grid(m, out_h, out_w, src_h, src_w):
    """Normalized grid_sample grid (align_corners=False) equivalent to
    cv2.warpAffine(src, M, (out_w, out_h)): dst(x,y) = src(inv(M) @ [x,y,1])."""
    full = np.vstack([m, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(full)[:2]
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float64)
    sx = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
    sy = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
    gx = (2.0 * sx + 1.0) / src_w - 1.0
    gy = (2.0 * sy + 1.0) / src_h - 1.0
    return np.stack([gx, gy], -1).astype(np.float32)


def dump_align(args, manifest):
    rng = np.random.default_rng(args.seed + 2)
    src_img = synth_image(320, 320, args.seed + 3, "face")

    # Two landmark sets: an upright one and a rotated+scaled one (a transposed or
    # x/y-swapped M is a no-op on the first and obvious on the second).
    lmk_a = np.array([[108.0, 140.0], [206.0, 138.0], [158.0, 196.0],
                      [118.0, 246.0], [198.0, 244.0]], dtype=np.float32)
    th = np.deg2rad(23.0)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    lmk_b = ((lmk_a - 160.0) @ rot.T * 1.35 + np.array([170.0, 150.0])).astype(np.float32)
    lmk_c = (lmk_a + rng.normal(0.0, 3.0, lmk_a.shape)).astype(np.float32)

    t = {"arcface_dst_112": ARCFACE_DST}
    meta = {}
    for tag, lmk in (("a", lmk_a), ("b", lmk_b), ("c", lmk_c)):
        m = estimate_norm(lmk, 112)
        m_ind = umeyama_similarity(lmk, ARCFACE_DST)
        d = float(np.abs(m - m_ind).max())
        assert d < 1e-4, f"skimage vs independent Umeyama differ by {d:.3e}"
        # M must be a true similarity: [[a,-b],[b,a]]
        a11, a12, a21, a22 = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
        assert abs(a11 - a22) < 1e-4 and abs(a12 + a21) < 1e-4, "M is not a similarity"
        scale = float(np.hypot(a11, a21))
        proj = (np.hstack([lmk, np.ones((5, 1))]) @ m.T)
        resid = float(np.abs(proj - ARCFACE_DST).max())

        warped = cv2.warpAffine(src_img, m, (112, 112), borderValue=0.0)
        grid = warp_grid(m, 112, 112, src_img.shape[0], src_img.shape[1])
        src_t = torch.from_numpy(src_img.astype(np.float32).transpose(2, 0, 1))[None]
        gs = torch.nn.functional.grid_sample(
            src_t, torch.from_numpy(grid)[None], mode="bilinear",
            padding_mode="zeros", align_corners=False)[0].numpy()
        gs_vs_cv2 = float(np.abs(gs.transpose(1, 2, 0) - warped.astype(np.float32)).max())

        t[f"lmk_{tag}"] = lmk
        t[f"M_{tag}"] = m.astype(np.float32)
        t[f"Minv_{tag}"] = np.linalg.inv(np.vstack([m, [0, 0, 1.0]]))[:2].astype(np.float32)
        t[f"grid_{tag}"] = grid
        t[f"warp_cv2_{tag}_u8"] = warped.astype(np.float32)
        t[f"warp_grid_sample_{tag}"] = gs
        t[f"blob_{tag}"] = arcface_blob(warped)[0]
        meta[tag] = {"scale": scale, "landmark_residual_max": resid,
                     "skimage_vs_umeyama_max": d,
                     "cv2_warpAffine_vs_torch_grid_sample_max_abs": gs_vs_cv2}
        print(f"  align[{tag}] scale {scale:.4f} residual {resid:.3e} "
              f"cv2-vs-grid_sample {gs_vs_cv2:.3f}/255", flush=True)
    t["src_img_u8"] = src_img.astype(np.float32)
    save(args.out, "align.safetensors", t, manifest, {
        "reference": "insightface.utils.face_align.{estimate_norm,norm_crop}, "
                     "skimage SimilarityTransform (Umeyama) + cv2.warpAffine",
        "M_convention": "2x3, dst = M @ [x, y, 1]; the warp samples "
                        "src at inv(M) @ [x_dst, y_dst, 1] (pixel centres at "
                        "integer coordinates)",
        "grid_convention": "grid_sample align_corners=False, padding_mode=zeros, "
                           "g = (2*src_coord + 1)/src_size - 1, layout (H, W, 2) "
                           "as (gx, gy)",
        "per_case": meta,
    })


# ---------------------------------------------------------------------------
# end-to-end on real photos
# ---------------------------------------------------------------------------
def dump_e2e(args, manifest, det_sess, det_in, det_outs, scratch):
    rec_path = os.path.join(args.weights, "glintr100.onnx")
    rec = ort.InferenceSession(rec_path, providers=["CPUExecutionProvider"])
    rec_in = rec.get_inputs()[0].name

    items, seen = [], set()
    for p in args.photos:
        if not os.path.exists(p):
            print(f"  photo missing, skipped: {p}", flush=True)
            continue
        digest = hashlib.md5(open(p, "rb").read()).hexdigest()
        if digest in seen:
            print(f"  duplicate file, skipped: {os.path.basename(p)}", flush=True)
            continue
        seen.add(digest)
        img = cv2.imread(p)
        if img is None:
            print(f"  photo unreadable, skipped: {p}", flush=True)
            continue
        items.append((os.path.basename(p), img))
    if items:
        # A deterministic re-capture of the FIRST identity (0.85x resize + a
        # brightness/contrast shift). Without it every pair is a different
        # person, so the cosine matrix cannot gate the same-identity direction.
        name, base = items[0]
        v = cv2.resize(base, (max(1, int(base.shape[1] * 0.85)),
                              max(1, int(base.shape[0] * 0.85))))
        v = np.clip(v.astype(np.float32) * 1.12 - 14.0, 0, 255).astype(np.uint8)
        items.append((f"{name}+variant(0.85x, 1.12*v-14)", v))

    t, embs, meta = {}, [], {}
    for i, (name, img) in enumerate(items):
        p = name
        det_img, det_scale, blob, _outs, _dec, dets, kpss = detect(
            det_sess, det_in, det_outs, img, 640)
        if dets.shape[0] == 0:
            print(f"  no face detected in {p}", flush=True)
            continue
        area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        j = int(np.argmax(area))
        kps = kpss[j].astype(np.float32)
        m = estimate_norm(kps, 112)
        aligned = cv2.warpAffine(img, m, (112, 112), borderValue=0.0)
        b = arcface_blob(aligned)
        emb = rec.run(None, {rec_in: b})[0].reshape(-1)
        tag = f"photo{i}"
        t[f"{tag}_det_img_u8"] = det_img.astype(np.float32)
        t[f"{tag}_det_blob"] = blob[0]
        t[f"{tag}_dets"] = dets
        t[f"{tag}_kpss"] = kpss
        t[f"{tag}_kps"] = kps
        t[f"{tag}_M"] = m.astype(np.float32)
        t[f"{tag}_aligned_bgr_u8"] = aligned.astype(np.float32)
        t[f"{tag}_blob"] = b[0]
        t[f"{tag}_embedding"] = emb
        t[f"{tag}_embedding_normed"] = emb / np.linalg.norm(emb)
        embs.append(emb / np.linalg.norm(emb))
        meta[tag] = {"file": name,
                     "src_hw": [int(img.shape[0]), int(img.shape[1])],
                     "det_scale": float(det_scale), "n_faces": int(dets.shape[0]),
                     "chosen_box": [float(x) for x in dets[j][:4]],
                     "score": float(dets[j][4]),
                     "embedding_l2_norm": float(np.linalg.norm(emb))}
        print(f"  {name}: {dets.shape[0]} face(s), "
              f"score {dets[j][4]:.4f}, |emb| {np.linalg.norm(emb):.3f}", flush=True)
    if not embs:
        print("  no usable photos - e2e.safetensors not written", flush=True)
        return
    e = np.stack(embs)
    t["cosine_matrix"] = e @ e.T
    print("  cosine matrix:\n" + "\n".join(
        "    " + " ".join(f"{v: .4f}" for v in row) for row in (e @ e.T)), flush=True)
    save(args.out, "e2e.safetensors", t, manifest, {
        "pipeline": "cv2.imread(BGR) -> SCRFD detect(640) -> largest box -> "
                    "estimate_norm(kps) -> cv2.warpAffine 112 -> blobFromImages "
                    "-> glintr100 -> 512-d embedding",
        "per_photo": meta,
        "cosine_matrix_order": [f"photo{i}" for i in range(len(embs))],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="antelopev2 directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--photos", nargs="*", default=[])
    ap.add_argument("--scratch", default=None,
                    help="where the tapped ONNX copies go (default: a temp dir, "
                         "removed on exit; they are regenerated on every run)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="arcface_taps_") if args.scratch is None else None
    scratch = args.scratch or tmp
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = {}
    print("== ArcFace / glintr100 (IResNet-100)", flush=True)
    dump_arcface(args, manifest, scratch)
    print("== 5-point alignment", flush=True)
    dump_align(args, manifest)
    if args.photos:
        print("== end-to-end on real photos (detect -> align -> embed)", flush=True)
        det_sess, det_in, det_outs = det_session(args.weights)
        dump_e2e(args, manifest, det_sess, det_in, det_outs, scratch)

    manifest["config"] = {
        "seed": args.seed,
        "weights_dir": os.path.abspath(args.weights),
        "models": {
            "recognition": "glintr100.onnx (ArcFace IResNet-100, 512-d, "
                           "BN folded into convs, opset 11, ir 6)",
            "detection": "scrfd_10g_bnkps.onnx (SCRFD-10GF, 9 outputs -> "
                         "fmc 3, strides [8,16,32], num_anchors 2, use_kps, "
                         "opset 11, ir 6) - run as shipped for the e2e chain "
                         "only; its stage goldens are scrfd_dump_reference.py's",
        },
        "arcface": {"input_mean": ARC_INPUT_MEAN, "input_std": ARC_INPUT_STD,
                    "input_size": [112, 112], "layout": "NCHW RGB",
                    "embedding_normalization": "none in-graph; consumers L2-"
                                               "normalize for cosine"},
        "scrfd": {"input_mean": DET_INPUT_MEAN, "input_std": DET_INPUT_STD,
                  "input_size": [640, 640], "layout": "NCHW RGB",
                  "det_thresh": DET_THRESH, "nms_thresh": NMS_THRESH,
                  "strides": FEAT_STRIDES, "num_anchors": NUM_ANCHORS},
        "align": {"template": "insightface arcface_dst (112x112)",
                  "solver": "skimage SimilarityTransform (Umeyama), "
                            "cross-checked against an independent solve",
                  "warp": "cv2.warpAffine bilinear, borderValue 0"},
        "versions": {"numpy": np.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__, "opencv": cv2.__version__,
                     "torch": torch.__version__,
                     "skimage": __import__("skimage").__version__,
                     "insightface_reference": "1.0.1 (source, vendored inline)"},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"done -> {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
