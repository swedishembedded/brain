#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Shared derivation of kernel metadata — the ONE implementation.

`gen-kernel-table.py` (which validates the declared `@` blocks) and
`seed-kernel-meta.py` (which proposes a block for a new kernel) both need the
same structural facts about a kernel. They each had their own copy, which is
exactly how one bug ended up in two places: both counted `workgroupBarrier`
over the RAW source, so both counted the word where a header *mentions* it, and
the seeder baked four wrong `@cpu` rows into the catalogue that the checker then
happily agreed with (`.agents/rules/lessons.md` #33).

Anything derived from a kernel's code belongs here, so a fix lands once and the
two scripts cannot disagree about what a kernel is.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
WGSL = ROOT / "crates" / "kernels" / "wgsl"

# Op families `crates/npu`'s ONNX topology DSL can emit. The NPU runs a whole
# compiled graph, never a WGSL kernel, so this says "expressible", not "runs".
NPU_EMITTABLE = (
    "matmul", "add", "mul", "sub", "reshape", "transpose", "softmax", "gather",
    "slice", "concat", "silu", "rmsnorm", "layernorm", "gn_", "group_norm",
    "conv", "linear_quant", "bn_", "relu", "gelu", "sigmoid", "tanh", "pool",
    "resize", "upsample", "embed", "pad", "crop",
)
QUANT_MARKERS = ("_i8", "i8_", "quant", "max_abs", "dequant", "bsq", "vq_")


def strip_comments(text):
    """Kernel body with `//` comments removed.

    Every structural fact below must be read from CODE. Reading the raw source
    counts a header that documents the kernel's own barrier discipline or
    workgroup size — and the kernels most likely to document those are exactly
    the ones the checks exist for, so the false positives land on the
    best-documented files (#33). WGSL block comments are not handled because no
    in-repo kernel uses them; add it here if one ever does.
    """
    return "\n".join(re.sub(r"//.*$", "", ln) for ln in text.splitlines())


def native_cpu_kernels():
    """Kernels with a hand-written AVX2 path, read from `backend_cpu`'s
    `FastIdx` so this cannot drift from the backend."""
    src = (ROOT / "crates" / "backend-cpu" / "src" / "lib.rs").read_text()
    block = re.search(r"struct FastIdx \{(.*?)\n\}", src, re.S)
    return set(re.findall(r"^\s{4}([a-z0-9_]+): Option<usize>", block.group(1), re.M)) if block else set()


def structure(name, text):
    """The structural facts, all read from code only."""
    code = strip_comments(text)
    wg = re.search(r"@workgroup_size\((\d+)", code)
    wg = int(wg.group(1)) if wg else 64
    # A loop is a serial reduction when its bound comes from the uniform —
    # directly (`d < p.head_dim`) or through the common local alias
    # (`let hd = p.head_dim;` then `d < hd`). Missing the aliased form
    # mis-classified every attention kernel.
    aliases = set(re.findall(r"let\s+(\w+)\s*(?::\s*\w+)?\s*=\s*p\.\w+", code))
    bound = r"p\.\w+" + ("|" + "|".join(map(re.escape, aliases)) if aliases else "")
    return {
        "wg": wg,
        "barriers": code.count("workgroupBarrier"),
        "shared": "var<workgroup>" in code,
        "dp4a": "dot4I8Packed" in code,
        "regblock": bool(re.search(r"_reg\d?$|_reg_", name)) or "rA[" in code,
        "loops": len(re.findall(r"for\s*\(\s*var\s+\w+[^;]*;\s*\w+\s*<\s*(?:%s)\b" % bound, code)),
    }


def how(name, st):
    tags = []
    if st["dp4a"]:
        tags.append("DP4A packed int8")
    if st["regblock"]:
        tags.append("register block per thread")
    if st["shared"]:
        tags.append(f"{st['wg']}-thread workgroup tile")
    if st["barriers"]:
        tags.append(f"{st['barriers']} barrier" + ("s" if st["barriers"] > 1 else ""))
    if not tags:
        tags.append("one thread per output element")
        if st["loops"] >= 3:
            tags.append(f"{st['loops']} nested serial reductions")
        elif st["loops"]:
            tags.append("serial inner reduction")
    return ", ".join(tags)


def opt(name, st):
    if st["dp4a"] or "splitk" in name or re.search(r"_reg\d?$", name):
        return 5
    if st["shared"] and st["barriers"]:
        return 4
    if name.endswith(("_part", "_final", "_rows", "_wg", "2")) and st["loops"]:
        return 4
    if st["loops"] >= 3:
        return 1
    if st["loops"]:
        return 2
    return 3


def cpu(name, st, native):
    """`yes` / `no` / `native` / `native-only`.

    The CPU JIT splits a kernel body at ONE top-level barrier and no more; with
    two or more it does not fail cleanly, it corrupts memory (#26).
    """
    if name in native:
        return "native" if st["barriers"] <= 1 else "native-only"
    return "yes" if st["barriers"] <= 1 else "no"


def gpu(st):
    return "yes-wg256" if st["wg"] >= 256 else "yes"


def npu(name):
    return "yes" if any(k in name for k in NPU_EMITTABLE) else "no"


def quant(name, st):
    # q4 (int4 weight, W4A8) kernels have no builtin to detect the way
    # `dot4I8Packed` marks int8 — they unpack nibbles by hand — so the name
    # marker is the signal, same convention `QUANT_MARKERS` already uses for
    # int8's non-dp4a members.
    if "q4" in name:
        return "q4"
    return "int8" if (st["dp4a"] or any(k in name for k in QUANT_MARKERS)) else "none"
