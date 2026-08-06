#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""ONE-TIME bootstrap: seed the `@`-tagged metadata block into every .wgsl.

The kernel catalogue in README.md is generated from metadata the kernels
*declare*, not from heuristics run over their code — a heuristic guesses, and a
guess in a reference table is worse than no table. This script writes the
initial blocks, seeded from the same structural signals the first version of
`gen-kernel-table.py` inferred plus each kernel's own prose; from then on the
block is the source of truth and authors maintain it like any other doc comment.

It is idempotent: a file that already has a `@what` tag is left alone, so
re-running it after adding kernels only seeds the new ones.

    scripts/build/seed-kernel-meta.py [--dry-run]
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
WGSL = ROOT / "crates" / "kernels" / "wgsl"

NPU_EMITTABLE = (
    "matmul", "add", "mul", "sub", "reshape", "transpose", "softmax", "gather",
    "slice", "concat", "silu", "rmsnorm", "layernorm", "gn_", "group_norm",
    "conv", "linear_quant", "bn_", "relu", "gelu", "sigmoid", "tanh", "pool",
    "resize", "upsample", "embed", "pad", "crop",
)
QUANT_MARKERS = ("_i8", "i8_", "quant", "max_abs", "dequant", "bsq", "vq_")


def native_cpu_kernels():
    src = (ROOT / "crates" / "backend-cpu" / "src" / "lib.rs").read_text()
    block = re.search(r"struct FastIdx \{(.*?)\n\}", src, re.S)
    return set(re.findall(r"^\s{4}([a-z0-9_]+): Option<usize>", block.group(1), re.M)) if block else set()


def strip_comments(text):
    """Kernel body with `//` comments removed.

    Counting `workgroupBarrier` over the raw source counts the word where a
    header *mentions* it — which is exactly what a kernel documenting its own
    barrier discipline does, so the cross-check fired on a correct kernel
    (`paged_decode_scores_wg`, one barrier, green on `backend-cpu`). A checker
    that cannot tell code from prose produces false positives on the kernels
    that document themselves best.
    """
    return "\n".join(re.sub(r"//.*$", "", ln) for ln in text.splitlines())


def header_lines(text):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("//"):
            out.append(s[2:].strip())
        elif not s:
            if out:
                break
        else:
            break
    return out


def summarise(lines):
    if not lines:
        return "TODO: one-line summary"
    buf = []
    for ln in lines:
        if not ln:
            break
        buf.append(ln)
        if ln.endswith(".") and len(" ".join(buf)) > 20:
            break
    s = " ".join(buf).strip()
    m = re.search(r"^(.{25,}?[.:])(\s|$)", s)
    if m:
        s = m.group(1)
    return (s.rstrip(".:").strip() or "TODO: one-line summary").replace("|", "/")


def derive(name, text):
    wg = re.search(r"@workgroup_size\((\d+)", text)
    wg = int(wg.group(1)) if wg else 64
    barriers = strip_comments(text).count("workgroupBarrier")
    shared = "var<workgroup>" in strip_comments(text)
    dp4a = "dot4I8Packed" in strip_comments(text)
    # A loop is a serial reduction when its bound comes from the uniform —
    # either directly (`d < p.head_dim`) or through the far more common local
    # alias (`let hd = p.head_dim;` then `d < hd`). Missing the aliased form
    # mis-seeded every attention kernel, including `gqa_scores`, which the
    # profile had already measured at 1.6% of the bandwidth roof.
    aliases = set(re.findall(r"let\s+(\w+)\s*(?::\s*\w+)?\s*=\s*p\.\w+", text))
    bound = r"p\.\w+" + ("|" + "|".join(map(re.escape, aliases)) if aliases else "")
    loops = len(re.findall(r"for\s*\(\s*var\s+\w+[^;]*;\s*\w+\s*<\s*(?:%s)\b" % bound, text))

    tags = []
    if dp4a:
        tags.append("DP4A packed int8")
    if re.search(r"_reg\d?$|_reg_", name) or "rA[" in text:
        tags.append("register block per thread")
    if shared:
        tags.append(f"{wg}-thread workgroup tile")
    if barriers:
        tags.append(f"{barriers} barrier" + ("s" if barriers > 1 else ""))
    if not tags:
        tags.append("one thread per output element")
        if loops >= 3:
            tags.append(f"{loops} nested serial reductions")
        elif loops:
            tags.append("serial inner reduction")
    how = ", ".join(tags)

    if dp4a or "splitk" in name or re.search(r"_reg\d?$", name):
        opt = 5
    elif shared and barriers:
        opt = 4
    elif name.endswith(("_part", "_final", "_rows", "_wg", "2")) and loops:
        opt = 4
    elif loops >= 3:
        opt = 1
    elif loops:
        opt = 2
    else:
        opt = 3
    return how, opt, barriers, wg, dp4a


def main():
    dry = "--dry-run" in sys.argv
    native = native_cpu_kernels()
    seeded = skipped = 0
    for f in sorted(WGSL.glob("*.wgsl")):
        text = f.read_text()
        if re.search(r"^//\s*@what\b", text, re.M):
            skipped += 1
            continue
        name = f.stem
        how, opt, barriers, wg, dp4a = derive(name, text)
        what = summarise(header_lines(text))
        cpu = ("native" if barriers <= 1 else "native-only") if name in native else ("yes" if barriers <= 1 else "no")
        gpu = "yes-wg256" if wg >= 256 else "yes"
        npu = "yes" if any(k in name for k in NPU_EMITTABLE) else "no"
        quant = "int8" if (dp4a or any(k in name for k in QUANT_MARKERS)) else "none"

        block = (
            f"// @what  {what}\n"
            f"// @how   {how}\n"
            f"// @opt   {opt}\n"
            f"// @cpu   {cpu}\n"
            f"// @gpu   {gpu}\n"
            f"// @npu   {npu}\n"
            f"// @quant {quant}\n"
            f"//\n"
        )
        if not dry:
            f.write_text(block + text)
        seeded += 1
    print(f"seeded {seeded}, already tagged {skipped}" + (" (dry run)" if dry else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
