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
    `return` within a few lines (a note that does not return is not a skip).
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


def count(path):
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
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
