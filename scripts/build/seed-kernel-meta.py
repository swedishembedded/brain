#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""ONE-TIME bootstrap: seed the `@`-tagged metadata block into every .wgsl.

The kernel catalogue is generated from metadata the kernels
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import kernelmeta as km  # noqa: E402  (path set above)

WGSL = km.WGSL


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


def main():
    dry = "--dry-run" in sys.argv
    native = km.native_cpu_kernels()
    seeded = skipped = 0
    for f in sorted(WGSL.glob("*.wgsl")):
        text = f.read_text()
        if re.search(r"^//\s*@what\b", text, re.M):
            skipped += 1
            continue
        name = f.stem
        st = km.structure(name, text)
        what = summarise(header_lines(text))

        block = (
            f"// @what  {what}\n"
            f"// @how   {km.how(name, st)}\n"
            f"// @opt   {km.opt(name, st)}\n"
            f"// @cpu   {km.cpu(name, st, native)}\n"
            f"// @gpu   {km.gpu(st)}\n"
            f"// @npu   {km.npu(name)}\n"
            f"// @quant {km.quant(name, st)}\n"
            f"//\n"
        )
        if not dry:
            f.write_text(block + text)
        seeded += 1
    print(f"seeded {seeded}, already tagged {skipped}" + (" (dry run)" if dry else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
