#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Drift guard for the model catalog, docs/reference/models/.

Checks two things, both cheap and weights-free:

1. Every model id `brain caps --json` reports (the capability-registry surface
   `crates/catalog/src/lib.rs` builds - what `brain <model>`/D-Bus/HTTP can
   actually reach) is named on a model reference page. This is the exact bug
   class `catalog.rs`'s module docs describe: a model wired into the registry
   and forgotten in the docs (or renamed in one place and not the other).
2. Every page the catalog index (docs/reference/models/index.md) links to
   exists.

Deliberately NOT a full set-equality check in the other direction: the pages
also document architectures with no weights-free `caps::manifest()` of their
own, which never appear in `brain caps` at all.

Usage: model_table_check.py --repo REPO --brain BRAIN_BIN
Exits 0 and prints "OK: ..." on success; prints each problem and exits 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Utility/demo entries that are real catalog models but not "a model" for the
# catalog's purposes (no ML weights, nothing to document a support matrix
# for). Weighted models NEVER belong here: `brain/qwen3vl` was once
# whitelisted with a "forward-only" rationale that was already false when it
# was written (commit 6b4e44a had landed a real served `generate` action), the
# exact drift this guard exists to catch. If a new model trips this check,
# name its id on its reference page; don't extend this set.
IGNORE = {"brain/mock", "brain/demo", "brain/imageops"}

# `brain caps` is a WEIGHTS-FREE static listing: it reports each catalog
# entry's crate-level `caps::MODEL` constant (`brain/qwen3`, `brain/lfm2`), which
# is a family placeholder, not a specific checkpoint — resolving to an actual
# HF ref (`Qwen/Qwen3-0.6B`, `LiquidAI/LFM2.5-350M`) only happens later, via
# `brain_modelstore`, which `brain caps` never touches. The pages document
# the ref a user would actually type (the auto-fetchable one), so these two
# are legitimate aliases, not documentation drift: the same relationship as
# `brain_modelref::alias` maps a legacy short name to its canonical id, just
# in the other direction (a family placeholder to its recommended checkpoint).
CATALOG_ALIASES = {
    "brain/qwen3": "Qwen/Qwen3-0.6B",
    "brain/lfm2": "LiquidAI/LFM2.5-350M",
    "brain/s3dit": "Tongyi-MAI/Z-Image-Turbo",
    "brain/yolov8": "Ultralytics/YOLOv8",
}

ROW_RE = re.compile(r"^\|\s*\[`([^`]+)`\]\(([^)]+)\)")


def index_rows(index: Path) -> list[tuple[str, str]]:
    """`(architecture, page)` for every row of every table in the catalog index."""
    if not index.is_file():
        print(f"FAIL: no model catalog at {index}", file=sys.stderr)
        sys.exit(1)
    return [mm.groups() for line in index.read_text().splitlines() if (mm := ROW_RE.match(line))]


def catalog_ids(brain: str) -> list[str]:
    out = subprocess.run([brain, "caps", "--json"], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        print(f"FAIL: `{brain} caps --json` exited {out.returncode}:\n{out.stderr}", file=sys.stderr)
        sys.exit(1)
    return [m["model"] for m in json.loads(out.stdout)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--brain", required=True)
    args = p.parse_args()

    models = args.repo / "docs" / "reference" / "models"
    rows = index_rows(models / "index.md")
    pages = [p.read_text() for p in models.rglob("*.md")]
    problems: list[str] = []

    for cid in catalog_ids(args.brain):
        if cid in IGNORE:
            continue
        wanted = CATALOG_ALIASES.get(cid, cid)
        if not any(wanted in page for page in pages):
            problems.append(f"'{cid}' is in `brain caps` but no page under docs/reference/models names '{wanted}'")

    for arch, link in rows:
        if not (models / link).is_file():
            problems.append(f"the catalog's '{arch}' row links to '{link}', which does not exist")

    if problems:
        print(f"FAIL: {len(problems)} problem(s):", file=sys.stderr)
        for prob in problems:
            print(f"  - {prob}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {len(rows)} catalog row(s), all links resolve, all `brain caps` ids documented")


if __name__ == "__main__":
    main()
