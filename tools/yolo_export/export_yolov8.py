#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# ruff: noqa: E501
"""Offline exporter: Ultralytics ``yolov8n.pt`` -> brain native ``.weights``.

PREREQUISITES (dev machine only — NOT this CI box, which has no torch/GPU):

    pip install ultralytics torch

USAGE:

    # 1. Export weights into brain's native container.
    python3 export_yolov8.py --weights yolov8n.pt --out yolov8n.brain.weights

    # 2. (Optional) Also dump per-stage activations for the parity test, for one
    #    fixed preprocessed 640x640 input image (role="act" tensors).
    python3 export_yolov8.py --weights yolov8n.pt --out yolov8n.brain.weights \
        --dump-acts --image bus.jpg --acts-out yolov8n.acts.weights

    # 3. Feed the two files to the Rust parity test (see crates/yolo/README.md):
    YOLO_PARITY_WEIGHTS=yolov8n.brain.weights \
    YOLO_PARITY_ACTS=yolov8n.acts.weights \
        cargo test -p brain-yolo --test parity -- --nocapture

WHAT THIS DOES
--------------
The brain YOLOv8 detector registers its parameters under names produced by
``YoloConfig::full_param_list`` (see crates/yolo/src/config.rs). This script
reads an official Ultralytics ``yolov8n.pt`` ``state_dict`` and maps each tensor
name onto the corresponding brain name via an EXPLICIT, auditable function
(:func:`ultra_to_brain`) — a 1:1 string remap with NO arithmetic on values.

Per Conv unit the mapping is:

    Ultralytics                          brain
    -----------                          -----
    <m>.conv.weight                  ->  <p>.conv.weight
    <m>.bn.weight                    ->  <p>.bn.gamma
    <m>.bn.bias                      ->  <p>.bn.beta
    <m>.bn.running_mean              ->  <p>.bn.run_mean
    <m>.bn.running_var               ->  <p>.bn.run_var

FOLD RULE (documented, applied implicitly, NOT a value op): BOTH sides keep the
convolution **bias-free** and carry the BatchNorm as four separate live tensors
(gamma/beta/run_mean/run_var). We do NOT fold BN into the conv weight — the
brain Conv runs conv2d (bias-free) -> BatchNorm -> SiLU exactly like
Ultralytics, so the BN tensors are copied straight across 1:1. ``num_batches_
tracked`` (a scalar bookkeeping tensor) is dropped — it has no brain counterpart.

OUTPUT FORMAT (byte-for-byte ``checkpoint::save``; see crates/checkpoint/src/lib.rs):

    [u64 LE json_header_len][json header bytes][f32 LE blob]

    header = {
      "config":  <YoloConfig::yolov8n().to_json()>,
      "tensors": [ {"name", "shape":[numel], "offset", "numel"}, ... ]   # role="weights" omitted
    }

    offset/numel are in **f32 units** (not bytes); the blob is the tensors
    concatenated in header order. We match brain's writer: tensors carry a flat
    1-element shape ``[numel]`` (model.rs `save` writes `vec![numel]`), no role
    field (role defaults to "" on read, which `Container::by_role("")` keys on).

BYTE-COMPATIBLE (P12): brain's ``YoloConfig::yolov8n()`` is now the EXACT
canonical Ultralytics yolov8n graph — per-stage channels
``[16,32,32,64,64,128,128,256,256,256]``, C2f depths ``[1,2,2,1]``, neck widths
``[128,64,64,128,128,256]``, and a BIASED decoupled head (``head.{s}.{cls,reg}.2``
has both ``.weight`` and ``.bias``). So a real ``yolov8n.pt`` maps 1:1 with equal
shapes. Two Ultralytics tensors are intentionally dropped (no brain counterpart):

  * ``...bn.num_batches_tracked`` — BN bookkeeping scalar.
  * ``model.22.dfl.conv.weight`` — the fixed DFL projection (arange ``16->1``).
    brain computes the DFL box expectation ANALYTICALLY, so it has no DFL conv.

The script still VALIDATES every mapped tensor's shape against the brain target
and FAILS LOUDLY (full per-tensor report) rather than writing a corrupt file —
now a guard rather than an expected failure.

The brain-side target names + element counts are checked in at
``brain_names.txt`` (regenerate with the p8_names ``dump_brain_names`` test).
With ``--check-map-only`` this script needs NO torch: it reconstructs the FULL
canonical yolov8n state_dict (name+shape) from the spec and proves the map covers
it 1:1 with equal shapes (see :func:`canonical_ultra_tensors` / :func:`check_map_only`).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

# ---------------------------------------------------------------------------
# Brain config (mirror of YoloConfig::yolov8n().to_json()). Kept in sync with
# crates/yolo/src/config.rs::yolov8n by hand — it is small and stable.
# ---------------------------------------------------------------------------
BRAIN_CONFIG = {
    "model": "yolo",
    "input": 640,
    "nc": 80,
    "reg_max": 16,
    "depth_mult": 0.33,
    "width_mult": 0.25,
    "channels": [16, 32, 64, 128, 256],
    "strides": [8, 16, 32],
    # Canonical yolov8n explicit layout (mirrors YoloConfig::yolov8n; see config.rs).
    "backbone_ch": [16, 32, 32, 64, 64, 128, 128, 256, 256, 256],
    "backbone_depth": [1, 2, 2, 1],
    "neck_ch": [128, 64, 64, 128, 128, 256],
    "neck_depth": 1,
    "cls_mid": 80,
    "reg_mid": 64,
}

# Ultralytics module-index -> brain prefix. yolov8n DetectionModel layout:
#   backbone 0..9, neck 12/15/16/18/19/21, head 22 (Detect).
# (Indices 10/13 are Upsample, 11/14/17/20 are Concat — they hold NO weights.)
ULTRA_INDEX_TO_BRAIN = {
    0: "backbone.0",   # Conv  3->16   s2
    1: "backbone.1",   # Conv 16->32   s2
    2: "backbone.2",   # C2f
    3: "backbone.3",   # Conv 32->64   s2
    4: "backbone.4",   # C2f   (P3)
    5: "backbone.5",   # Conv 64->128  s2
    6: "backbone.6",   # C2f   (P4)
    7: "backbone.7",   # Conv 128->256 s2
    8: "backbone.8",   # C2f
    9: "backbone.9",   # SPPF  (P5)
    12: "neck.0",      # C2f   top-down P4
    15: "neck.1",      # C2f   top-down P3 (= N3, head scale 0)
    16: "neck.2",      # Conv  downsample N3
    18: "neck.3",      # C2f   (= N4, head scale 1)
    19: "neck.4",      # Conv  downsample N4
    21: "neck.5",      # C2f   (= N5, head scale 2)
    # 22 (Detect head) handled specially in ultra_to_brain.
}

# Ultralytics BN suffix -> brain BN suffix (1:1, no value math).
BN_SUFFIX = {
    "bn.weight": "bn.gamma",
    "bn.bias": "bn.beta",
    "bn.running_mean": "bn.run_mean",
    "bn.running_var": "bn.run_var",
}

# Ultralytics Detect head: cv2 = box(reg), cv3 = cls. Each is a Sequential
# [Conv, Conv, Conv2d]; brain calls these reg/cls with the same .0/.1/.2 indices.
HEAD_BRANCH = {"cv2": "reg", "cv3": "cls"}


def _remap_conv_tail(tail: str) -> str | None:
    """Map the part of an Ultralytics name AFTER a Conv-unit prefix.

    ``tail`` is e.g. "conv.weight" / "bn.weight" / "bn.running_mean". Returns the
    brain tail, or None to drop (e.g. ``num_batches_tracked``).
    """
    if tail == "conv.weight":
        return "conv.weight"
    if tail in BN_SUFFIX:
        return BN_SUFFIX[tail]
    if tail.endswith("num_batches_tracked"):
        return None
    return None  # anything else (e.g. an unexpected bias) is unmapped


def ultra_to_brain(name: str) -> str | None:
    """Map one Ultralytics ``state_dict`` key to its brain name, or None if it
    has no brain counterpart (and must be dropped / reported as unmatched).

    Pure string remap; never touches tensor values. Every branch is explicit so
    the mapping is auditable line-by-line.
    """
    if not name.startswith("model."):
        return None
    parts = name.split(".")
    idx = int(parts[1])
    rest = parts[2:]  # tokens after "model.<idx>."

    # --- Detect head (model.22): cv2/cv3 . {scale} . {0,1,2} . ... -----------
    if idx == 22:
        # Ultralytics Detect submodules: cv2 (box), cv3 (cls), dfl (fixed).
        if rest and rest[0] in HEAD_BRANCH:
            branch = HEAD_BRANCH[rest[0]]  # reg | cls
            scale = rest[1]                # "0" | "1" | "2"
            sub = rest[2]                  # "0" | "1" | "2" (layer in the Sequential)
            tail = ".".join(rest[3:])      # "conv.weight" | "bn.*" | "weight" | "bias"
            base = f"head.{scale}.{branch}.{sub}"
            if sub in ("0", "1"):
                # full Conv (conv+BN+SiLU)
                bt = _remap_conv_tail(tail)
                return f"{base}.{bt}" if bt else None
            if sub == "2":
                # final layer: Ultralytics biased Conv2d (weight + bias); brain's
                # head is now ALSO biased (P12), so BOTH map 1:1.
                if tail == "weight":
                    return f"{base}.weight"
                if tail == "bias":
                    return f"{base}.bias"
                return None
            return None
        # model.22.dfl.conv.weight (fixed DFL projection) — no brain counterpart.
        return None

    # --- backbone / neck modules --------------------------------------------
    if idx not in ULTRA_INDEX_TO_BRAIN:
        return None  # Upsample/Concat (no params) or an unexpected index
    base = ULTRA_INDEX_TO_BRAIN[idx]

    # Plain Conv module (model.0/1/3/5/7/16/19): tail = conv.weight | bn.*
    # C2f/SPPF: tail starts with cv1/cv2/m.<i>.cv<j> then conv.weight|bn.*
    tail = ".".join(rest)
    bt = _remap_conv_tail(tail)
    if bt is not None and "." not in tail.rsplit(".", 1)[0] or bt is not None:
        # tail is exactly a Conv tail (plain Conv module)
        # (the condition above is just a guard; _remap_conv_tail handles it)
        # but only valid if the WHOLE tail is a conv/bn tail.
        if tail in ("conv.weight",) or tail in BN_SUFFIX or tail.endswith("num_batches_tracked"):
            return f"{base}.{bt}" if bt else None

    # Nested (C2f / SPPF): peel sub-prefixes (cv1 / cv2 / m.<i>.cv<j>) until the
    # remaining tail is a Conv tail. brain uses the SAME cv1/cv2/m.<i> names.
    toks = rest
    sub_prefix_toks: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("cv1", "cv2"):
            sub_prefix_toks.append(t)
            i += 1
        elif t == "m":
            # m.<i>
            sub_prefix_toks += [t, toks[i + 1]]
            i += 2
        else:
            break
    sub_prefix = ".".join(sub_prefix_toks)
    conv_tail = ".".join(toks[i:])
    bt = _remap_conv_tail(conv_tail)
    if bt is None:
        return None
    if sub_prefix:
        return f"{base}.{sub_prefix}.{bt}"
    return f"{base}.{bt}"


# ---------------------------------------------------------------------------
# brain .weights writer (reimplements checkpoint::save in pure Python).
# ---------------------------------------------------------------------------
def write_brain_weights(path: str, config: dict, tensors: list[tuple[str, list[float], str]]) -> None:
    """tensors: list of (name, flat_f32_values, role). role="" matches the
    brain writer (model.rs save) which omits the field entirely; we emit it only
    when non-empty so weight files stay byte-identical to ``checkpoint::save``."""
    entries = []
    blob = bytearray()
    offset = 0  # f32 units
    for name, vals, role in tensors:
        numel = len(vals)
        entry = {"name": name, "shape": [numel], "offset": offset, "numel": numel}
        if role:
            entry["role"] = role
        entries.append(entry)
        blob += struct.pack(f"<{numel}f", *vals)
        offset += numel
    header = {"config": config, "tensors": entries}
    # serde_json default separators (", " / ": "); we don't need byte-identical
    # JSON, only a valid header serde_json::from_str can parse. Compact is fine.
    hbytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        f.write(blob)


# ---------------------------------------------------------------------------
# brain-side name table (checked-in dump of full_param_list()).
# ---------------------------------------------------------------------------
def load_brain_names(path: str) -> list[tuple[str, int]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, numel = line.rsplit(" ", 1)
            out.append((name, int(numel)))
    return out


# ---------------------------------------------------------------------------
# Canonical Ultralytics yolov8n state_dict, derived from the documented spec
# WITHOUT torch. Every shape is computed from the channel/depth numbers:
#   * conv.weight   -> [out, in, kh, kw]
#   * bn.{weight,bias,running_mean,running_var} -> [C]
#   * bn.num_batches_tracked -> []  (scalar; dropped)
#   * head final Conv2d -> weight [out, mid, 1, 1], bias [out]
#   * dfl.conv.weight -> [1, reg_max, 1, 1]  (fixed; dropped)
# This is the SAME layout YoloConfig::yolov8n() encodes on the brain side, so a
# real yolov8n.pt's state_dict (name+shape) equals this list 1:1.
# ---------------------------------------------------------------------------
# yolov8n canonical numbers (width_mult 0.25, depth_mult 0.33, max_channels 1024).
_BCH = [16, 32, 32, 64, 64, 128, 128, 256, 256, 256]  # backbone stage out-channels
_BDEPTH = [1, 2, 2, 1]                                  # C2f depths for stages 2,4,6,8
_NECK = [128, 64, 64, 128, 128, 256]                    # neck.0..5 out-channels
_NDEPTH = 1
_NC, _REG_MAX = 80, 16
_CLS_MID, _REG_MID = 80, 64
# Backbone stage index -> module index in the Ultralytics nn.Sequential.
_BACKBONE_MODULE = {i: i for i in range(10)}  # backbone.i == model.i for 0..9
# Neck stage -> module index: neck.0=12, neck.1=15, neck.2=16, neck.3=18,
# neck.4=19, neck.5=21.
_NECK_MODULE = [12, 15, 16, 18, 19, 21]


def _u_conv(mod: int, sub_prefix: str, cin: int, cout: int, k: int) -> list[tuple[str, list[int]]]:
    """Ultralytics Conv unit (conv2d bias-free + BN) tensors with shapes."""
    p = f"model.{mod}" + (f".{sub_prefix}" if sub_prefix else "")
    return [
        (f"{p}.conv.weight", [cout, cin, k, k]),
        (f"{p}.bn.weight", [cout]),
        (f"{p}.bn.bias", [cout]),
        (f"{p}.bn.running_mean", [cout]),
        (f"{p}.bn.running_var", [cout]),
        (f"{p}.bn.num_batches_tracked", []),  # dropped on export
    ]


def _u_c2f(mod: int, cin: int, cout: int, n: int) -> list[tuple[str, list[int]]]:
    c = cout // 2
    out: list[tuple[str, list[int]]] = []
    out += _u_conv(mod, "cv1", cin, 2 * c, 1)
    prev = c
    for i in range(n):
        out += _u_conv(mod, f"m.{i}.cv1", prev, c, 3)
        out += _u_conv(mod, f"m.{i}.cv2", c, c, 3)
        prev = c
    out += _u_conv(mod, "cv2", (2 + n) * c, cout, 1)
    return out


def _u_sppf(mod: int, cin: int, cout: int) -> list[tuple[str, list[int]]]:
    c = cout // 2
    return _u_conv(mod, "cv1", cin, c, 1) + _u_conv(mod, "cv2", 4 * c, cout, 1)


def canonical_ultra_tensors() -> list[tuple[str, list[int]]]:
    """The full canonical Ultralytics yolov8n state_dict as (name, shape), torch-free."""
    t: list[tuple[str, list[int]]] = []
    # backbone 0..9
    t += _u_conv(0, "", 3, _BCH[0], 3)
    t += _u_conv(1, "", _BCH[0], _BCH[1], 3)
    t += _u_c2f(2, _BCH[1], _BCH[2], _BDEPTH[0])
    t += _u_conv(3, "", _BCH[2], _BCH[3], 3)
    t += _u_c2f(4, _BCH[3], _BCH[4], _BDEPTH[1])
    t += _u_conv(5, "", _BCH[4], _BCH[5], 3)
    t += _u_c2f(6, _BCH[5], _BCH[6], _BDEPTH[2])
    t += _u_conv(7, "", _BCH[6], _BCH[7], 3)
    t += _u_c2f(8, _BCH[7], _BCH[8], _BDEPTH[3])
    t += _u_sppf(9, _BCH[8], _BCH[9])
    p3, p4, p5 = _BCH[4], _BCH[6], _BCH[9]
    # neck (module indices per _NECK_MODULE)
    t += _u_c2f(_NECK_MODULE[0], p5 + p4, _NECK[0], _NDEPTH)
    t += _u_c2f(_NECK_MODULE[1], _NECK[0] + p3, _NECK[1], _NDEPTH)
    t += _u_conv(_NECK_MODULE[2], "", _NECK[1], _NECK[2], 3)
    t += _u_c2f(_NECK_MODULE[3], _NECK[2] + _NECK[0], _NECK[3], _NDEPTH)
    t += _u_conv(_NECK_MODULE[4], "", _NECK[3], _NECK[4], 3)
    t += _u_c2f(_NECK_MODULE[5], _NECK[4] + p5, _NECK[5], _NDEPTH)
    # head (model.22): cv2 = reg/box (mid=_REG_MID, out=4*reg_max),
    #                  cv3 = cls (mid=_CLS_MID, out=nc). 3 scales on [N3,N4,N5].
    head_in = [_NECK[1], _NECK[3], _NECK[5]]
    reg_out, cls_out = 4 * _REG_MAX, _NC
    for s, cin in enumerate(head_in):
        # cv2 (reg/box branch)
        t += _u_conv(22, f"cv2.{s}.0", cin, _REG_MID, 3)
        t += _u_conv(22, f"cv2.{s}.1", _REG_MID, _REG_MID, 3)
        t.append((f"model.22.cv2.{s}.2.weight", [reg_out, _REG_MID, 1, 1]))
        t.append((f"model.22.cv2.{s}.2.bias", [reg_out]))
        # cv3 (cls branch)
        t += _u_conv(22, f"cv3.{s}.0", cin, _CLS_MID, 3)
        t += _u_conv(22, f"cv3.{s}.1", _CLS_MID, _CLS_MID, 3)
        t.append((f"model.22.cv3.{s}.2.weight", [cls_out, _CLS_MID, 1, 1]))
        t.append((f"model.22.cv3.{s}.2.bias", [cls_out]))
    # dfl projection (fixed; dropped on export)
    t.append((f"model.22.dfl.conv.weight", [1, _REG_MAX, 1, 1]))
    return t


# Ultralytics tensors that are intentionally dropped (no brain counterpart).
def _is_dropped_ultra(name: str) -> bool:
    return name.endswith("num_batches_tracked") or name == "model.22.dfl.conv.weight"


def check_map_only(brain_names_path: str) -> int:
    """Torch-free self-test PROVING the exporter loads a real yolov8n.pt 1:1.

    We synthesize the canonical Ultralytics yolov8n state_dict (name + shape) from
    the documented spec (:func:`canonical_ultra_tensors`) — no torch needed, every
    shape is arithmetic on the channel/depth numbers. Then we assert, against the
    checked-in brain target list (`brain_names.txt`, the dump of
    `YoloConfig::yolov8n().full_param_list()`):

      (a) every NON-dropped Ultralytics tensor maps to exactly one brain name,
      (b) the mapped brain tensor's element count EQUALS prod(canonical shape),
      (c) every brain name is covered by exactly one Ultralytics tensor,
      (d) every dropped Ultralytics tensor (num_batches_tracked, dfl.conv.weight)
          indeed maps to None.

    Since a real `yolov8n.pt`'s state_dict has exactly these names+shapes, passing
    this proves the export will map+shape-check 1:1 on a dev machine with torch."""
    brain = load_brain_names(brain_names_path)
    brain_numel = {n: c for n, c in brain}
    brain_set = set(brain_numel)

    ultra = canonical_ultra_tensors()
    problems: list[str] = []
    covered: dict[str, str] = {}  # brain name -> ultra key that covered it

    for uname, ushape in ultra:
        b = ultra_to_brain(uname)
        if _is_dropped_ultra(uname):
            if b is not None:
                problems.append(f"dropped tensor {uname!r} unexpectedly mapped to {b!r}")
            continue
        if b is None:
            problems.append(f"ultra tensor {uname!r} -> None (expected a brain name)")
            continue
        if b not in brain_set:
            problems.append(f"ultra tensor {uname!r} -> {b!r} (not a brain name)")
            continue
        if b in covered:
            problems.append(f"brain name {b!r} mapped by TWO ultra tensors: {covered[b]!r} and {uname!r}")
            continue
        want = 1
        for d in ushape:
            want *= d
        if brain_numel[b] != want:
            problems.append(f"shape mismatch {uname!r} -> {b!r}: ultra numel {want} != brain numel {brain_numel[b]}")
            continue
        covered[b] = uname

    missing = brain_set - set(covered)
    if missing:
        problems.append(f"{len(missing)} brain names NOT covered, e.g. {sorted(missing)[:8]}")

    if problems:
        print("MAP CHECK FAILED:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        return 1
    n_ultra = len(ultra)
    n_dropped = sum(1 for n, _ in ultra if _is_dropped_ultra(n))
    print(
        f"MAP CHECK OK: canonical yolov8n state_dict = {n_ultra} tensors "
        f"({n_dropped} dropped: num_batches_tracked + dfl.conv.weight); "
        f"remaining {n_ultra - n_dropped} map 1:1 onto all {len(brain_set)} brain "
        f"tensors with EQUAL shapes. The exporter will load a real yolov8n.pt 1:1."
    )
    return 0


def synth_ultra_keys(brain_set: set[str]) -> list[str]:
    """Invert the documented brain->Ultralytics naming to produce the canonical
    Ultralytics key for each brain tensor name. Pure string transform."""
    inv_index = {v: k for k, v in ULTRA_INDEX_TO_BRAIN.items()}
    inv_bn = {v: k for k, v in BN_SUFFIX.items()}  # brain bn -> ultra bn
    inv_branch = {v: k for k, v in HEAD_BRANCH.items()}  # reg->cv2, cls->cv3
    keys = []
    for name in sorted(brain_set):
        if name.startswith("head."):
            # head.<scale>.<branch>.<sub>.<tail>
            _, scale, branch, sub, *tail_parts = name.split(".")
            cv = inv_branch[branch]
            tail = ".".join(tail_parts)
            if sub in ("0", "1"):
                u_tail = _brain_conv_tail_to_ultra(tail, inv_bn)
                keys.append(f"model.22.{cv}.{scale}.{sub}.{u_tail}")
            else:  # sub == "2": ".weight" or ".bias" (head is biased, P12)
                u_tail = tail  # "weight" | "bias" — 1:1 on both sides
                keys.append(f"model.22.{cv}.{scale}.{sub}.{u_tail}")
            continue
        # backbone.<i>... / neck.<i>...
        toks = name.split(".")
        base = f"{toks[0]}.{toks[1]}"  # e.g. backbone.4
        idx = inv_index[base]
        rest = toks[2:]
        # split off the conv tail (last 2 toks for conv.weight/bn.*; bn.* is 2 toks)
        # find where the conv tail starts: it's the trailing conv.weight or bn.<x>
        if rest[-2:] == ["conv", "weight"]:
            sub_prefix = ".".join(rest[:-2])
            u_tail = "conv.weight"
        else:
            # bn.<gamma|beta|run_mean|run_var>
            sub_prefix = ".".join(rest[:-2])
            bn_brain = ".".join(rest[-2:])
            u_tail = inv_bn[bn_brain]
        if sub_prefix:
            keys.append(f"model.{idx}.{sub_prefix}.{u_tail}")
        else:
            keys.append(f"model.{idx}.{u_tail}")
    return keys


def _brain_conv_tail_to_ultra(tail: str, inv_bn: dict[str, str]) -> str:
    if tail == "conv.weight":
        return "conv.weight"
    return inv_bn[tail]  # bn.gamma -> bn.weight, etc.


# ---------------------------------------------------------------------------
# Full export (requires torch + ultralytics).
# ---------------------------------------------------------------------------
def export(weights: str, out: str, brain_names_path: str | None) -> int:
    import torch  # noqa: F401  (defer; only needed for the real export)
    from ultralytics import YOLO

    m = YOLO(weights)
    sd = m.model.state_dict()

    brain_numel = {}
    if brain_names_path and os.path.exists(brain_names_path):
        brain_numel = {n: c for n, c in load_brain_names(brain_names_path)}

    mapped: dict[str, list[float]] = {}
    unmatched = []
    shape_mismatch = []
    for k, v in sd.items():
        b = ultra_to_brain(k)
        if b is None:
            unmatched.append(k)
            continue
        flat = v.detach().cpu().contiguous().view(-1).float().tolist()
        if b in brain_numel and len(flat) != brain_numel[b]:
            shape_mismatch.append((k, b, len(flat), brain_numel[b]))
        mapped[b] = flat

    # Report coverage against the brain target list, in brain (graph) order.
    order = [n for n, _ in load_brain_names(brain_names_path)] if brain_names_path else sorted(mapped)
    missing = [n for n in order if n not in mapped]

    print(f"mapped {len(mapped)} tensors; {len(unmatched)} Ultralytics tensors unmatched.")
    if unmatched:
        print("  unmatched (dropped) Ultralytics tensors, e.g.:", unmatched[:12])
    if missing:
        print(f"  WARNING: {len(missing)} brain tensors have NO source, e.g.: {missing[:12]}")
    if shape_mismatch:
        print(f"  ERROR: {len(shape_mismatch)} shape mismatches (brain graph != yolov8n):")
        for k, b, got, want in shape_mismatch[:20]:
            print(f"    {k} -> {b}: got {got} f32, brain wants {want}")
        print("  Refusing to write a corrupt file. See crates/yolo/README.md 'Discrepancies'.")
        return 2
    if missing:
        print("  Refusing to write an incomplete file (some brain tensors unmapped).")
        return 3

    tensors = [(n, mapped[n], "") for n in order]
    write_brain_weights(out, BRAIN_CONFIG, tensors)
    print(f"wrote {out} ({len(tensors)} tensors).")
    return 0


# ---------------------------------------------------------------------------
# Optional activation dump (requires torch + ultralytics).
# ---------------------------------------------------------------------------
def dump_acts(weights: str, image: str, acts_out: str) -> int:
    import numpy as np  # noqa: F401
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    m = YOLO(weights)
    net = m.model.eval()

    # Preprocess one image to a fixed 640x640 CHW float tensor in [0,1].
    import cv2

    img = cv2.imread(image)
    if img is None:
        print(f"cannot read image {image}", file=sys.stderr)
        return 1
    lb = LetterBox((640, 640), auto=False)
    img = lb(image=img)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    x = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0) / 255.0

    # Register forward hooks on each weighted module index that maps to a brain
    # stage, capturing the module OUTPUT activation under the brain stage name.
    captured: list[tuple[str, list[float]]] = []
    handles = []

    def make_hook(brain_name):
        def hook(_mod, _inp, output):
            t = output[0] if isinstance(output, (list, tuple)) else output
            captured.append((brain_name, t.detach().cpu().contiguous().view(-1).float().tolist()))
        return hook

    seq = net.model  # nn.Sequential of the 23 modules
    for idx, brain_name in ULTRA_INDEX_TO_BRAIN.items():
        handles.append(seq[idx].register_forward_hook(make_hook(brain_name)))

    with torch.no_grad():
        net(x)
    for h in handles:
        h.remove()

    tensors = [(name, vals, "act") for name, vals in captured]
    # Also dump the input image itself so the Rust test uploads the identical
    # preprocessed tensor (name "input", role "act").
    tensors.insert(0, ("input", x.contiguous().view(-1).float().tolist(), "act"))
    write_brain_weights(acts_out, BRAIN_CONFIG, tensors)
    print(f"wrote {acts_out} ({len(tensors)} activation tensors).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolov8n.pt", help="Ultralytics .pt checkpoint")
    ap.add_argument("--out", default="yolov8n.brain.weights", help="output brain .weights path")
    ap.add_argument(
        "--brain-names",
        default=os.path.join(os.path.dirname(__file__), "brain_names.txt"),
        help="checked-in dump of YoloConfig::yolov8n().full_param_list() names",
    )
    ap.add_argument("--check-map-only", action="store_true", help="torch-free: verify the name map covers all brain names")
    ap.add_argument("--dump-acts", action="store_true", help="also dump per-stage activations")
    ap.add_argument("--image", default="bus.jpg", help="image for the activation dump")
    ap.add_argument("--acts-out", default="yolov8n.acts.weights", help="activation dump output path")
    args = ap.parse_args()

    if args.check_map_only:
        return check_map_only(args.brain_names)

    rc = export(args.weights, args.out, args.brain_names)
    if rc != 0:
        return rc
    if args.dump_acts:
        rc = dump_acts(args.weights, args.image, args.acts_out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
