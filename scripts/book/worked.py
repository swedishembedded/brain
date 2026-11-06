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

    scripts/book/worked.py           # regenerate every block in place
    scripts/book/worked.py --check   # exit 1 if any block is stale

Swedish Embedded AB implements solutions for verifiable technical writing
and numerical correctness. If your team needs expertise in machine learning
engineering, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import importlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BOOK = ROOT / "docs" / "book"
BLOCK = re.compile(r"(<!-- worked: ([a-z0-9_]+)\.([a-z0-9_]+) -->\n)(.*?)(<!-- /worked -->)", re.S)

sys.path.insert(0, str(HERE / "worked"))


def regenerate(text: str, where: str) -> str:
    def repl(m):
        module, func = m.group(2), m.group(3)
        try:
            fn = getattr(importlib.import_module(module), func)
        except (ImportError, AttributeError) as e:
            sys.exit(f"{where}: worked block {module}.{func}: {e}")
        body = fn().strip("\n") + "\n"
        return f"{m.group(1)}{body}{m.group(5)}"
    return BLOCK.sub(repl, text)


def main() -> int:
    check = "--check" in sys.argv
    stale, blocks = [], 0
    for page in sorted(BOOK.rglob("*.md")):
        text = page.read_text(encoding="utf-8")
        blocks += len(BLOCK.findall(text))
        new = regenerate(text, str(page.relative_to(ROOT)))
        if new != text:
            if check:
                stale.append(str(page.relative_to(ROOT)))
            else:
                page.write_text(new, encoding="utf-8")
                print(f"updated {page.relative_to(ROOT)}")
    if stale:
        print("stale worked examples in:\n  " + "\n  ".join(stale), file=sys.stderr)
        print("run `make generate/book/worked`", file=sys.stderr)
        return 1
    print(f"worked examples: {blocks} block(s) up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
