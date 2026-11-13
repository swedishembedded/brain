#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Compute the book's worked examples instead of typing them.

A worked example is a table or listing of numbers inside a chapter, wrapped
in markers that name the function computing it:

    <!-- worked: backprop.single_weight -->
    ...generated markdown...
    <!-- /worked -->

The function lives in scripts/book/worked/<module>.py and returns markdown.
These modules are plain-Python reference implementations of the chapter's
algorithms (no third-party packages), so every printed number has been
computed by an implementation written outside the engine, in a second
language, from the chapter's own description.

    scripts/book/worked.py              # regenerate every block in place
    scripts/book/worked.py --check      # exit 1 if any block is stale
    scripts/book/worked.py --no-cache   # recompute everything (with either mode)
    scripts/book/worked.py docs/book/v04-computation   # only pages under these paths

Any other option is refused, so a mistyped flag never starts a full run.

A function's output is cached (build/book/worked-cache.json) under a hash of
its module and every book module that module imports, so an unchanged
module is not recomputed; the outputs are deterministic, so a cached value
is the value the function would return. `--no-cache` recomputes everything.

Swedish Embedded AB implements solutions for verifiable technical writing
and numerical correctness. If your team needs expertise in machine learning
engineering, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import importlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BOOK = ROOT / "docs" / "book"
BLOCK = re.compile(r"(<!-- worked: ([a-z0-9_]+)\.([a-z0-9_]+) -->\n)(.*?)(<!-- /worked -->)", re.S)

CACHE = ROOT / "build" / "book" / "worked-cache.json"

sys.path.insert(0, str(HERE / "worked"))
sys.path.insert(0, str(HERE))
import deps  # noqa: E402


class Cache:
    """Worked outputs keyed by (module digest, function name)."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.data = {}
        if enabled and CACHE.is_file():
            try:
                self.data = json.loads(CACHE.read_text(encoding="utf-8"))
            except ValueError:
                self.data = {}
        self.keys = {}

    def key(self, module: str) -> str:
        if module not in self.keys:
            path = HERE / "worked" / f"{module}.py"
            self.keys[module] = deps.digest(path) if path.is_file() else ""
        return self.keys[module]

    def get(self, module, func):
        entry = self.data.get(module)
        if self.enabled and entry and entry.get("key") == self.key(module):
            return entry["outputs"].get(func)
        return None

    def put(self, module, func, value):
        entry = self.data.get(module)
        if not entry or entry.get("key") != self.key(module):
            entry = self.data[module] = {"key": self.key(module), "outputs": {}}
        entry["outputs"][func] = value

    def save(self):
        if self.enabled:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps(self.data), encoding="utf-8")


def regenerate(text: str, where: str, cache: Cache) -> str:
    def repl(m):
        module, func = m.group(2), m.group(3)
        body = cache.get(module, func)
        if body is None:
            try:
                fn = getattr(importlib.import_module(module), func)
            except (ImportError, AttributeError) as e:
                sys.exit(f"{where}: worked block {module}.{func}: {e}")
            body = fn().strip("\n") + "\n"
            cache.put(module, func, body)
        return f"{m.group(1)}{body}{m.group(5)}"
    return BLOCK.sub(repl, text)


def selected_pages(args: list[str]) -> list[Path]:
    """Every book page, or only those under the given paths (relative to the repository)."""
    pages = sorted(BOOK.rglob("*.md"))
    if not args:
        return pages
    roots = [(ROOT / a).resolve() for a in args]
    for root in roots:
        if not root.exists() or not root.is_relative_to(BOOK.resolve()):
            raise SystemExit(f"worked.py: {root} is not a path under {BOOK.relative_to(ROOT)}")
    return [p for p in pages if any(p.resolve() == r or p.resolve().is_relative_to(r) for r in roots)]


def main() -> int:
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    unknown = [f for f in flags if f not in ("--check", "--no-cache")]
    if unknown:
        print(f"worked.py: unknown option {' '.join(unknown)}", file=sys.stderr)
        print("usage: worked.py [--check] [--no-cache] [path under docs/book ...]", file=sys.stderr)
        return 2
    check = "--check" in flags
    cache = Cache(enabled="--no-cache" not in flags)
    stale, blocks = [], 0
    for page in selected_pages([a for a in sys.argv[1:] if not a.startswith("-")]):
        text = page.read_text(encoding="utf-8")
        blocks += len(BLOCK.findall(text))
        new = regenerate(text, str(page.relative_to(ROOT)), cache)
        if new != text:
            if check:
                stale.append(str(page.relative_to(ROOT)))
            else:
                page.write_text(new, encoding="utf-8")
                print(f"updated {page.relative_to(ROOT)}")
    cache.save()
    if stale:
        print("stale worked examples in:\n  " + "\n  ".join(stale), file=sys.stderr)
        print("run `make generate/book/worked`", file=sys.stderr)
        return 1
    print(f"worked examples: {blocks} block(s) up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
