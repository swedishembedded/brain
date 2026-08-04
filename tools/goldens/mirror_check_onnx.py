#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Verify exported WorldMirror-2 ONNX graphs under OpenVINO.

Runs each graph on the deterministic T2/T4 inputs (same as the Rust parity
tests) and compares against the committed golden samples in
crates/mirror/tests/golden/golden_meta.json.

  python3 tools/goldens/mirror_check_onnx.py out/mirror-dino.onnx  [CPU|NPU]
  python3 tools/goldens/mirror_check_onnx.py out/mirror-trunk.onnx [CPU|NPU] --stage trunk
      (trunk input = crates/mirror/tests/golden/t2_patch_tokens.npy from the
       dump script; taps compared vs the t4_tap goldens — export with
       --frames 1 --hp 37 --wp 37)
  python3 tools/goldens/mirror_check_onnx.py out [CPU|NPU] --stage heads
      (chains out/mirror-trunk.onnx -> out/mirror-{depth,pts,norm,gs}_head.onnx
       and compares every head + gs_params against the t5 goldens)
"""
import json
import sys

import numpy as np
import openvino as ov
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def synth_image(w=600, h=400):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x, 0] = (x * 255) // max(w - 1, 1)
            img[y, x, 1] = (y * 255) // max(h - 1, 1)
            img[y, x, 2] = ((x * 7 + y * 13) // 4) % 256
    yy, xx = np.mgrid[0:h, 0:w]
    for (cx, cy, r, col) in [(150, 100, 60, (255, 40, 40)), (420, 260, 90, (30, 220, 90)), (300, 180, 30, (250, 250, 250))]:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[m] = col
    return img


# The trunk taps come out as separate frame/global halves: emitting them as
# Concat(frame, global) straight into a graph output miscompiled on the Intel
# NPU (the global half came back holding the frame half's values, 13.8% rms
# error at the last level) while the identical tensor routed out on its own
# was correct to 0.18%. See crates/npu/src/mirror_topology.rs.
def check_trunk(path, device):
    core = ov.Core()
    model = core.compile_model(path, device)
    toks = np.load("crates/mirror/tests/golden/t2_patch_tokens.npy")  # [1369,1024]
    out = model({"patch_tokens": toks[None]})
    meta = json.load(open("crates/mirror/tests/golden/golden_meta.json"))
    fail = False
    for i in range(4):
        # the graph emits the frame/global halves separately; concatenate here
        tap = np.concatenate([out[f"tap{i}_frame"][0], out[f"tap{i}_global"][0]], axis=-1)
        s = meta[f"t4_tap{i}"]
        flat = tap.reshape(-1)
        rms = float(np.sqrt(np.mean(flat.astype(np.float64) ** 2)))
        worst = max(abs(float(flat[j]) - v) for j, v in zip(s["indices"], s["values"]))
        # NPU is fp16-only; drift grows with depth (measured median relative
        # error 5e-4 at tap0 rising to 2.5e-2 at tap3 after 48 attention
        # blocks, per-row cosine >= 0.9990, rms within 0.55%).
        rms_tol, tol = (0.02, 2e-1) if device == "NPU" else (0.002, 3e-3)
        print(f"tap{i}: rms {rms:.6f} (golden {s['rms']:.6f}), worst sampled abs diff {worst:.2e}")
        if abs(rms - s["rms"]) > rms_tol * abs(s["rms"]) or worst > tol:
            fail = True
    if fail:
        print("MISMATCH")
        sys.exit(1)
    print("OK — trunk ONNX matches the reference within tolerance")


def check_heads(outdir, device):
    core = ov.Core()
    trunk = core.compile_model(f"{outdir}/mirror-trunk.onnx", device)
    toks = np.load("crates/mirror/tests/golden/t2_patch_tokens.npy")
    taps = trunk({"patch_tokens": toks[None]})
    meta = json.load(open("crates/mirror/tests/golden/golden_meta.json"))
    # head inputs = the PATCH rows (skip the 7 special tokens)
    feed = {
        f"tap{i}": np.concatenate(
            [taps[f"tap{i}_frame"], taps[f"tap{i}_global"]], axis=-1
        )[:, 7:, :]
        for i in range(4)
    }
    pil = Image.fromarray(synth_image(), "RGB")
    sq = pil.crop((41, 0, 441, 400)).resize((518, 518), Image.Resampling.BICUBIC)
    rgb = (np.asarray(sq).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    fail = False

    def cmp(key, flat, tol):
        nonlocal fail
        s = meta[key]
        rms = float(np.sqrt(np.mean(flat.astype(np.float64) ** 2)))
        worst = max(abs(float(flat[j]) - v) for j, v in zip(s["indices"], s["values"]))
        print(f"{key}: rms {rms:.6f} (golden {s['rms']:.6f}), worst sampled abs diff {worst:.2e}")
        if abs(rms - s["rms"]) > (0.02 if device == "NPU" else 0.005) * abs(s["rms"]) or worst > tol:
            fail = True

    tol = 2e-1 if device == "NPU" else 5e-3
    for name, key in [("depth_head", "t5_depth_head"), ("pts_head", "t5_pts_head"), ("norm_head", "t5_norm_head")]:
        m = core.compile_model(f"{outdir}/mirror-{name}.onnx", device)
        cmp(key, m(feed)["head_out"].reshape(-1), tol)
    m = core.compile_model(f"{outdir}/mirror-gs_head.onnx", device)
    gout = m({**feed, "rgb": rgb})
    cmp("t5_gs_head", gout["head_out"].reshape(-1), tol)
    cmp("t5_gs_params", gout["gs_params"].reshape(-1), tol)
    if fail:
        print("MISMATCH")
        sys.exit(1)
    print("OK — all four DPT heads match the reference within tolerance")


def main():
    args = [a for a in sys.argv[1:] if a != "--stage"]
    stage = "trunk" if "trunk" in sys.argv[1:] and "--stage" in sys.argv[1:] else "dino"
    args = [a for a in args if a != "trunk"]
    path = args[0] if args else "out/mirror-dino.onnx"
    device = args[1] if len(args) > 1 else "CPU"
    stage = "heads" if "heads" in sys.argv[1:] and "--stage" in sys.argv[1:] else stage
    args = [a for a in args if a != "heads"]
    path = args[0] if args else "out/mirror-dino.onnx"
    device = args[1] if len(args) > 1 else "CPU"
    if stage == "trunk":
        check_trunk(path, device)
        return
    if stage == "heads":
        check_heads(path if path != "out/mirror-dino.onnx" else "out", device)
        return
    pil = Image.fromarray(synth_image(), "RGB")
    sq = pil.crop((41, 0, 441, 400)).resize((518, 518), Image.Resampling.BICUBIC)
    sn = ((np.asarray(sq).astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    x = sn.transpose(2, 0, 1)[None]  # [1,3,518,518]

    core = ov.Core()
    model = core.compile_model(path, device)
    out = model({"frame": x})["patch_tokens"][0]  # [1369,1024]

    meta = json.load(open("crates/mirror/tests/golden/golden_meta.json"))
    s = meta["t2_patch_tokens"]
    flat = out.reshape(-1)
    rms = float(np.sqrt(np.mean(flat.astype(np.float64) ** 2)))
    ok = abs(rms - s["rms"]) < 0.001 * abs(s["rms"])
    worst = 0.0
    for i, v in zip(s["indices"], s["values"]):
        worst = max(worst, abs(float(flat[i]) - v))
    # The Intel NPU executes fp16 only (it rejects INFERENCE_PRECISION_HINT
    # f32), so a 24-block residual stream accumulates ~2-3x fp16 eps of
    # relative error — measured median 1.3e-3 on significant values, with
    # per-token cosine similarity >= 0.99985 vs the fp32 CPU run. Gate the
    # NPU on that reality instead of the fp32 tolerance.
    tol = 5e-2 if device == "NPU" else 5e-4
    rms_tol = 0.01 if device == "NPU" else 0.001
    ok = abs(rms - s["rms"]) < rms_tol * abs(s["rms"])
    print(f"device {device}: rms {rms:.6f} (golden {s['rms']:.6f}), worst sampled abs diff {worst:.2e}")
    if not ok or worst > tol:
        print("MISMATCH")
        sys.exit(1)
    print("OK — ONNX export matches the reference within tolerance")


if __name__ == "__main__":
    main()
