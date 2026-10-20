#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for keeping large Rust inference and
# training stacks layered and buildable for its clients. If your team needs
# expertise in architectural dependency enforcement then you can procure our
# services by sending an email to info@swedishembedded.com.
"""Per-file pattern ratchet shared by the layering gates (G2-G5).

Counts production-code matches of one or more regexes per file and compares
them with an allowlist of `path<TAB>count<TAB>reason` lines. Fails on
  - a file with more matches than listed (a new violation: fix the design, an
    allowlist addition needs a reason and a note to the integrator);
  - a listed count above the real one, or for a file with none, or a missing
    reason (the list only shrinks).
Production code means crates/*/src/**/*.rs without line comments, and without
everything from the first column-0 `#[cfg(test)]` onwards. tests/, benches/
and examples/ are never scanned.
"""
import argparse
import re
import sys
from pathlib import Path


def package_of(crate_dir):
    toml = crate_dir / "Cargo.toml"
    if toml.exists():
        m = re.search(r'^name\s*=\s*"([^"]+)"', toml.read_text(), re.M)
        if m:
            return m.group(1)
    return None


def layer_map(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, layer = line.split("\t")[:2]
            out[name] = int(layer)
    return out


def production_text(path):
    lines = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#[cfg(test)]"):
            break
        if line.lstrip().startswith("//"):
            continue
        lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--allow", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pattern", action="append", required=True)
    ap.add_argument("--exclude", action="append", default=[], help="regex on the repo-relative path")
    ap.add_argument("--layers", help="comma-separated layers; only crates on them are scanned")
    ap.add_argument("--map")
    ap.add_argument("--advice", default="")
    ap.add_argument("--emit", action="store_true", help="print path<TAB>count for the current tree and exit")
    a = ap.parse_args()

    root = Path(a.root)
    pats = [re.compile(p) for p in a.pattern]
    excl = [re.compile(p) for p in a.exclude]
    layers = {int(x) for x in a.layers.split(",")} if a.layers else None
    lmap = layer_map(a.map or Path(__file__).resolve().parent / "crate-layers.map") if layers else {}

    actual = {}
    for crate in sorted((root / "crates").glob("*")):
        if layers is not None and lmap.get(package_of(crate)) not in layers:
            continue
        for f in sorted((crate / "src").rglob("*.rs")) if (crate / "src").is_dir() else []:
            rel = f.relative_to(root).as_posix()
            if any(e.search(rel) for e in excl):
                continue
            n = sum(1 for line in production_text(f) if any(p.search(line) for p in pats))
            if n:
                actual[rel] = n
    if a.emit:
        for rel, n in actual.items():
            print(f"{rel}\t{n}")
        return 0

    allow = {}
    problems = []
    for i, line in enumerate(Path(a.allow).read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3 or not parts[1].isdigit() or not parts[2].strip():
            problems.append(f"BAD ENTRY {Path(a.allow).name}:{i}: need path<TAB>count<TAB>reason")
            continue
        allow[parts[0]] = int(parts[1])
    for rel, n in sorted(actual.items()):
        have = allow.get(rel, 0)
        if n > have:
            problems.append(f"NEW {a.name}: {rel} has {n} match(es), {have} allowed. {a.advice}")
    for rel, have in sorted(allow.items()):
        n = actual.get(rel, 0)
        if have > n:
            problems.append(f"STALE ALLOWLIST ENTRY: {rel} lists {have} but has {n}; lower or delete the entry (the list only shrinks)")
    if problems:
        print("\n".join(problems))
        print(f"{a.name}: FAIL ({len(problems)} problem(s))")
        return 1
    print(f"{a.name}: OK, {sum(actual.values())} allowlisted match(es) in {len(actual)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
