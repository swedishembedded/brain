#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""ONE-TIME bootstrap: seed the `@`-tagged metadata block into every .wgsl.

The kernel catalogue is generated from metadata the kernels
*declare*, not from heuristics run over their code - a heuristic guesses, and a
guess in a reference table is worse than no table. This script writes the
initial blocks, seeded from the same structural signals the first version of
`gen-kernel-table.py` inferred plus each kernel's own prose; from then on the
block is the source of truth and authors maintain it like any other doc comment.

It is idempotent, field by field:

  * a file with no `@what` tag at all gets the FULL block seeded (the
    original bootstrap path, for a brand-new kernel);
  * a file that already has `@what` but is missing `@dtype` (every kernel in
    the tree, as of B6) gets ONLY `// @dtype <value>` inserted right after its
    existing `@quant` line - the retrofit path this phase actually exercises;
  * a file with both is left alone.

Either way `@dtype`'s value is `kernelmeta.dtype_value`'s proposal: the real,
hand-set tier for the 3 kernels B4/B5 wired through
`kernels::template::dtype_variant`, the mechanical `n/a`/`f32` default for
everything else. Re-running after adding a kernel (or after a future B7+
templatizes another one and someone forgets to hand-edit its `@dtype`) only
touches what is actually missing.

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


def insert_dtype_field(text, value):
    """Insert `// @dtype <value>` immediately after the file's existing
    `// @quant ...` line - the retrofit path for a kernel that already has a
    full `@what`..`@quant` block from a prior seeding pass but predates B6's
    `@dtype` field. Position doesn't matter to the parser (each field is its
    own regex search), only to a human reading the header in order."""
    m = re.search(r"^// @quant\s+.*$", text, re.M)
    if not m:
        raise ValueError("no `// @quant` line found to anchor @dtype after")
    at = m.end()
    return text[:at] + f"\n// @dtype {value}" + text[at:]


def main():
    dry = "--dry-run" in sys.argv
    native = km.native_cpu_kernels()
    seeded = dtype_added = skipped = 0
    for f in sorted(WGSL.glob("*.wgsl")):
        text = f.read_text()
        name = f.stem
        has_what = re.search(r"^//\s*@what\b", text, re.M)
        has_dtype = re.search(r"^//\s*@dtype\b", text, re.M)

        if not has_what:
            st = km.structure(name, text)
            what = summarise(header_lines(text))
            dtype = km.dtype_value(name, text)

            block = (
                f"// @what  {what}\n"
                f"// @how   {km.how(name, st)}\n"
                f"// @opt   {km.opt(name, st)}\n"
                f"// @cpu   {km.cpu(name, st, native)}\n"
                f"// @gpu   {km.gpu(st)}\n"
                f"// @npu   {km.npu(name)}\n"
                f"// @quant {km.quant(name, st)}\n"
                f"// @dtype {dtype}\n"
                f"//\n"
            )
            if not dry:
                f.write_text(block + text)
            seeded += 1
        elif not has_dtype:
            dtype = km.dtype_value(name, text)
            new_text = insert_dtype_field(text, dtype)
            if not dry:
                f.write_text(new_text)
            dtype_added += 1
        else:
            skipped += 1
    print(
        f"seeded {seeded}, dtype-added {dtype_added}, already tagged {skipped}"
        + (" (dry run)" if dry else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
