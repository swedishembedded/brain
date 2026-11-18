#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for test suites whose green result
# can be trusted for its clients. If your team needs expertise in test-gate
# design and continuous verification then you can procure our services by
# sending an email to info@swedishembedded.com.
"""Count the silent skips under crates/: `path<TAB>count`, sorted.

A silent skip is a test that returns green without having run its check:
  - a raw `std::env::var("MOE_SKIP_GPU_TESTS")` read (the early-return gate);
    `brain_testutil::gpu_tests_skipped` is the one place allowed to read it;
  - an `eprintln!("... skipping ...")` / `skipped` that is followed by a
    `return` within a few lines (a note that does not return is not a skip);
  - in a file under a `tests/` directory, a function that opens a real GPU
    (`Gpu::new_gpu`, `Gpu::new_wgpu`, `WgpuBackend::new`, ...) with no device
    guard before the open (`brain_testutil::gpu_unavailable`,
    `testgpu::missing_gpu`, `adapter_available`, `try_new_vulkan`, ..., or a
    helper of the same file that calls one): on a host without an adapter it
    panics instead of skipping by name. An `#[ignore]`d function (a benchmark)
    and a `card_tests!` binary (its card is a command-line argument) are exempt.
Comment lines never count.
"""
import argparse
import os
import re
import sys

ENV_GATE = re.compile(r'env::var(?:_os)?\(\s*"MOE_SKIP_GPU_TESTS"')
BARE_NOTE = re.compile(r'eprintln!\(\s*"[^"]*\bskipp(?:ing|ed)\b', re.I)
RETURN = re.compile(r'\breturn\b')
LOOKAHEAD = 3
EXEMPT_CRATE = "crates/testutil/"
DEVICE_OPEN = re.compile(r"\bGpu::new_(?:gpu|wgpu|wgpu_multi)\s*\(|\bWgpuBackend::new(?:_multi)?\s*\(")
DEVICE_GUARD = re.compile(
    r"\b(?:gpu_unavailable|missing_gpu|adapter_available|enumerate_gpus|try_new_vulkan|try_new_cuda)\s*\(|\b(?:testgpu::cards|VulkanBackend::try_new)\s*\("
)
FN = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")


def functions(lines):
    """`(name, ignored, body)` for every `fn` of `lines`, its body the text between its braces."""
    text = "\n".join(lines)
    out = []
    for m in FN.finditer(text):
        open_at = text.find("{", m.end())
        semi = text.find(";", m.end())
        if open_at < 0 or (0 <= semi < open_at):
            continue  # a declaration without a body
        depth, i = 0, open_at
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        head = text[: m.start()].splitlines()
        attrs = []
        for line in reversed(head):
            stripped = line.strip()
            if stripped.startswith("#[") or stripped.startswith("///") or stripped == "":
                attrs.append(stripped)
                continue
            break
        out.append((m.group(1), any(a.startswith("#[ignore") for a in attrs), text[open_at:i]))
    return out


def unguarded_opens(lines):
    """Functions that open a real GPU with no device guard before the open."""
    if any("card_tests!(" in line for line in lines):
        return 0
    fns = functions(lines)
    guards = {name for name, _, body in fns if DEVICE_GUARD.search(body)}
    helper = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(guards))) + r")\s*\(") if guards else None
    n = 0
    for _, ignored, body in fns:
        opened = DEVICE_OPEN.search(body)
        if ignored or not opened:
            continue
        before = body[: opened.start()]
        if DEVICE_GUARD.search(before) or (helper and helper.search(before)):
            continue
        n += 1
    return n


def count(path):
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    code = [line for line in lines if not line.lstrip().startswith("//")]
    if f"{os.sep}tests{os.sep}" in path:
        n += unguarded_opens(code)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("//"):
            continue
        if ENV_GATE.search(line):
            n += 1
        elif BARE_NOTE.search(line):
            window = lines[i : i + 1 + LOOKAHEAD]
            if any(RETURN.search(w) and not w.lstrip().startswith("//") for w in window):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    rows = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(args.root, "crates")):
        dirnames[:] = [d for d in dirnames if d not in ("target", ".git")]
        for f in filenames:
            if not f.endswith(".rs"):
                continue
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, args.root)
            if rel.startswith(EXEMPT_CRATE):
                continue
            n = count(full)
            if n:
                rows.append((rel, n))
    for rel, n in sorted(rows):
        print(f"{rel}\t{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
