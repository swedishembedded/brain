#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""List the process-environment writes in non-test code as `path<TAB>count`.

A write is a call of `env::set_var` or `env::remove_var`. The environment is
process-global state: a library that sets a variable around a build and restores
it afterwards lets every other thread of the process read the temporary value,
so two concurrent trainings of one seed once ran with different optimisers.

Not scanned, because a test owns its own process: files under `tests/`,
`benches/` or `examples/`, files named `tests.rs` or `*_tests.rs`, and any
`#[cfg(test)]` item with a braced body. Full-line `//` comments are ignored.
A binary's `main` is scanned like everything else; an entry for it on the
allowlist says why it owns the process.

usage: env-writes.py [--root DIR] [GLOB ...]
"""
import argparse
import re
import sys
from pathlib import Path

WRITE = re.compile(r"\benv::(?:set_var|remove_var)\s*\(")
CFG_TEST = re.compile(r"#\[cfg\(test\)\]")
TEST_DIRS = {"tests", "benches", "examples"}


def is_test_file(rel: Path) -> bool:
    name = rel.name
    return bool(TEST_DIRS & set(rel.parts[:-1])) or name == "tests.rs" or name.endswith("_tests.rs")


def writes_in(text: str) -> int:
    count = 0
    skip_until_depth = None  # brace depth at which the #[cfg(test)] item closes
    depth = 0
    armed = False  # a #[cfg(test)] attribute was seen; the next `{` opens its body
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue
        if skip_until_depth is None and CFG_TEST.search(line):
            armed = True
        if skip_until_depth is None and not armed:
            count += len(WRITE.findall(line))
        for ch in line:
            if ch == "{":
                if armed and skip_until_depth is None:
                    skip_until_depth = depth
                    armed = False
                depth += 1
            elif ch == "}":
                depth -= 1
                if skip_until_depth is not None and depth == skip_until_depth:
                    skip_until_depth = None
        if armed and ";" in line and "{" not in line:
            armed = False  # `#[cfg(test)] use ...;` has no body to skip
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("globs", nargs="*", default=["crates/**/*.rs"])
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for pattern in args.globs:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if is_test_file(rel):
                continue
            n = writes_in(path.read_text(encoding="utf-8", errors="replace"))
            if n:
                rows.append((rel.as_posix(), n))
    for rel, n in sorted(set(rows)):
        print(f"{rel}\t{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
