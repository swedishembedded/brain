#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Shared derivation of kernel metadata - the ONE implementation.

`gen-kernel-table.py` (which validates the declared `@` blocks) and
`seed-kernel-meta.py` (which proposes a block for a new kernel) both need the
same structural facts about a kernel. They each had their own copy, which is
exactly how one bug ended up in two places: both counted `workgroupBarrier`
over the RAW source, so both counted the word where a header *mentions* it, and
the seeder baked four wrong `@cpu` rows into the catalogue that the checker then
happily agreed with.

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
    workgroup size - and the kernels most likely to document those are exactly
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
    # A loop is a serial reduction when its bound comes from the uniform -
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
    # `dot4I8Packed` marks int8 - they unpack nibbles by hand - so the name
    # marker is the signal, same convention `QUANT_MARKERS` already uses for
    # int8's non-dp4a members.
    if "q4" in name:
        return "q4"
    return "int8" if (st["dp4a"] or any(k in name for k in QUANT_MARKERS)) else "none"


# --- @dtype (B6): does this kernel support bf16/f16 WEIGHT STORAGE? ---------
#
# `f32`   - the default: no float storage binding worth templatizing (either
#           none at all, or a tiny gain/bias vector where bf16/f16 storage
#           would be a numerically-questionable, VRAM-irrelevant optimisation
#           - norms, per-channel affine params).
# `n/a`   - literally no storage binding that could ever be templatized: every
#           storage binding is already `array<u32>` (pure index/scan/sort
#           kernels). Auto-verified below, not trusted.
# `f32|bf16` / `f32|bf16|f16` - a real storage tier, wired through
#           `kernels::template::dtype_variant` (B4/B5). Also auto-verified: a
#           binding must actually be declared `array<f32>` AND every load of
#           it must already index with a bare identifier - the templater's
#           hard precondition (`rewrite_packed_loads` in
#           `crates/kernels/src/template.rs`).

DTYPE_VALUES = ("f32", "n/a", "f32|bf16", "f32|bf16|f16")

# The kernels actually wired through `kernels::template::dtype_variant` - a
# fact about what `crates/kernels/src/template.rs` is wired for, not
# something derivable from the kernel's own source text, so it is recorded by
# hand instead of guessed at. B4/B5: the 3 matmul-family kernels. B8: the
# highest-value remaining weight-consuming inference kernels - embedding
# table gather (`embed`/`embed_tile`), the sparse-MoE expert linear
# (`moe_linear_gated`), and the 3 forward conv kernels whose own `w_idx` was
# already a bare identifier (`conv2d`/`conv1d`/`conv_bias` - see B8's ledger
# entry for why the register-tiled/grouped-dilated/workgroup-staged conv
# siblings were deliberately left `f32` instead).
DTYPE_TEMPLATIZED = {
    "matmul": "f32|bf16|f16",
    "matmul_gemv": "f32|bf16|f16",
    "matmul_reg3": "f32|bf16|f16",
    "embed": "f32|bf16|f16",
    "embed_tile": "f32|bf16|f16",
    "moe_linear_gated": "f32|bf16|f16",
    "conv2d": "f32|bf16|f16",
    "conv1d": "f32|bf16|f16",
    "conv_bias": "f32|bf16|f16",
}

_STORAGE_DECL = re.compile(r"var<storage,\s*(?:read|read_write)>\s+(\w+)\s*:\s*array<(\w+)>")
_BARE_IDENT = re.compile(r"^[A-Za-z_]\w*$")


def storage_bindings(text):
    """`[(binding_name, element_type), ...]` for every storage binding, read
    from CODE (comments stripped). Every kernel in this tree declares its
    storage bindings as either `array<f32>` or `array<u32>` - confirmed by
    grepping the whole `wgsl/` tree - so those are the only two element types
    this needs to distinguish."""
    return re.findall(_STORAGE_DECL, strip_comments(text))


def has_f32_storage_binding(text):
    return any(elem == "f32" for _, elem in storage_bindings(text))


def templatable_bindings(text):
    """Storage bindings declared `array<f32>` where EVERY `<binding>[...]`
    load already indexes with a bare identifier -
    `kernels::template::dtype_variant`'s hard precondition (B4/B5's
    `rewrite_packed_loads`: a compound index would be double-evaluated by the
    decode expansion). A binding with a compound index anywhere, or one never
    loaded at all, does not count."""
    code = strip_comments(text)
    out = []
    for name, elem in storage_bindings(text):
        if elem != "f32":
            continue
        loads = re.findall(r"(?<![A-Za-z0-9_])%s\[([^\[\]]*)\]" % re.escape(name), code)
        if loads and all(_BARE_IDENT.match(idx.strip()) for idx in loads):
            out.append(name)
    return out


def dtype_default(text):
    """The mechanical default for a kernel that is not one of the 3 hand-set
    templatized ones: `n/a` if it has zero `array<f32>` storage bindings
    (pure index/scan/sort kernels), `f32` otherwise."""
    return "n/a" if not has_f32_storage_binding(text) else "f32"


def dtype_value(name, text):
    """The `@dtype` value `seed-kernel-meta.py` proposes: the real hand-set
    value for the 3 kernels B4/B5 templatized, the mechanical default for
    everything else."""
    return DTYPE_TEMPLATIZED.get(name, dtype_default(text))


def dtype_errors(name, text, value):
    """Validate a DECLARED `@dtype` value against the code - the same
    "declaration must agree with the code" contract `gen-kernel-table.py`'s
    other mechanical cross-checks (`@cpu`/`@gpu`/`@quant`/`@opt 5`) already
    enforce. Kept here, not in `gen-kernel-table.py`, so this module's own
    direct self-check (`python3 scripts/build/kernelmeta.py`) and the
    catalogue's drift check can never disagree about what a kernel's `@dtype`
    should be."""
    if value not in DTYPE_VALUES:
        return [f"@dtype {value!r} is not one of {DTYPE_VALUES}"]
    if value == "n/a":
        if has_f32_storage_binding(text):
            return [
                "@dtype n/a but the kernel has a storage binding declared `array<f32>` - "
                "n/a means ZERO float storage bindings (pure index/scan/sort kernels with "
                "only array<u32> bindings), not merely 'not templatized'; did you mean "
                "@dtype f32?"
            ]
        return []
    if value != "f32":
        # f32|bf16[|f16]: claims a real storage tier beyond the f32 default.
        if not templatable_bindings(text):
            return [
                f"@dtype {value!r} claims a bf16/f16 storage tier but no storage binding is "
                "both declared `array<f32>` and indexed only by a bare identifier "
                "(kernels::template::dtype_variant's hard precondition, B4/B5) - either add "
                "the bare-identifier hoist the templater needs, or declare @dtype f32 instead"
            ]
        return []
    return []


def _self_check():
    """Scan every kernel's DECLARED `@dtype` (if any) and report every
    `dtype_errors` hit directly - a standalone way to exercise this module's
    own validation logic without going through `gen-kernel-table.py`. Skips
    kernels with no `@dtype` line at all (`gen-kernel-table.py`'s missing-field
    check owns that case, not this one)."""
    files = sorted(WGSL.glob("*.wgsl"))
    problems = []
    for f in files:
        text = f.read_text()
        m = re.search(r"^//\s*@dtype\s+(.*?)\s*$", text, re.M)
        if not m:
            continue
        for e in dtype_errors(f.stem, text, m.group(1)):
            problems.append(f"{f.stem}.wgsl: {e}")
    if problems:
        print("kernelmeta @dtype validation failed:")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"kernelmeta @dtype validation: {len(files)} kernel(s) scanned, 0 problems")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_self_check())
