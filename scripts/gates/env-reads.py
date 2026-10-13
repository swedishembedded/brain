#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""List every environment read under crates/ as `path<TAB>VARIABLE` lines.

The single source of truth for "what counts as an environment read", shared by
check-env-docs.sh (which documents each BRAIN_* name) and check-env-reads.sh
(the ratchet that forbids new reads). Two detectors, both ignoring full-line
`//` comments:

  1. any call whose first argument is a `"BRAIN_*"` literal - `env::var`,
     `var_os`, and helpers such as `env_path`/`env_or`/`host_env`;
  2. `env::var`/`env::var_os`/`env::vars`, `env!` and `option_env!` with any
     other argument: a literal name is reported as is, a computed name as
     `<expression>` (so a new indirect read in an already-listed file is
     still a new line). Cargo-provided compile-time variables (`CARGO_*`,
     `OUT_DIR`) are not user input and are skipped.

usage: env-reads.py [--root DIR] [GLOB ...]
  --root DIR  tree to scan (default: the repository root)
  GLOB        path globs relative to the root (default: crates/**/*.rs)
"""
import argparse
import re
import sys
from pathlib import Path

BRAIN_CALL = re.compile(r'\(\s*"(BRAIN_[A-Z0-9_]+)"')
STD_READ = re.compile(r'\benv::(?:var|var_os)\s*\(\s*("([^"]*)"|[^)]*)')
STD_VARS = re.compile(r'\benv::vars(?:_os)?\s*\(')
BUILD_READ = re.compile(r'\b(?:option_)?env!\s*\(\s*"([^"]*)"')
COMPILE_TIME_BUILTIN = re.compile(r'^(CARGO_.*|OUT_DIR)$')


def reads_in(text: str):
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue
        found = set(BRAIN_CALL.findall(line))
        for m in STD_READ.finditer(line):
            found.add(m.group(2) if m.group(2) is not None else "<" + m.group(1).strip() + ">")
        if STD_VARS.search(line):
            found.add("<all>")
        found.update(n for n in BUILD_READ.findall(line) if not COMPILE_TIME_BUILTIN.match(n))
        yield from found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("globs", nargs="*", default=["crates/**/*.rs"])
    args = ap.parse_args()
    root = Path(args.root)
    rows = set()
    for pattern in args.globs:
        for path in root.glob(pattern):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                text = path.read_text(encoding="utf-8", errors="replace")
                rows.update((rel, name) for name in reads_in(text))
    for rel, name in sorted(rows):
        sys.stdout.write(f"{rel}\t{name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
