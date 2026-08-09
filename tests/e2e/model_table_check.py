#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Drift guard for README.md's "## Model support" table.

Checks two things, both cheap and weights-free:

1. Every model id `brain caps --json` reports (the capability-registry surface
   `crates/cli/src/catalog.rs` builds — what `brain do`/D-Bus/HTTP can actually
   reach) appears somewhere in the table. This is the exact bug class
   `catalog.rs`'s module docs describe: a model wired into the registry and
   forgotten in the docs (or renamed in one place and not the other).
2. Every doc link the table points at resolves to a real file on disk.

Deliberately NOT a full set-equality check in the other direction: the table
also documents CLI-only / residency-only models (gpt, glm, moe, pid, nemotron,
qwen-asr, chronos2/fincast/kronos, mirror, splat, the DIAMOND world model) that
have no weights-free `caps::manifest()` and so never appear in `brain caps` at
all — asserting the reverse would fail on models that are correctly documented.

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
# support table's purposes (no ML weights, nothing to document a support
# matrix for). `brain/qwenvl` is a real, weighted VLM but forward-only (no
# serving action beyond raw forward) — README.md's own prose right above the
# table says it (and moondream, not yet in `brain caps`) is documented below
# the table instead of given a row; this mirrors that deliberate exclusion.
IGNORE = {"brain/mock", "brain/demo", "brain/imageops", "brain/qwenvl"}

# `brain caps` is a WEIGHTS-FREE static listing: it reports each catalog
# entry's crate-level `caps::MODEL` constant (`brain/qwen`, `brain/lfm`), which
# is a family placeholder, not a specific checkpoint — resolving to an actual
# HF ref (`Qwen/Qwen3-0.6B`, `LiquidAI/LFM2.5-350M`) only happens later, via
# `brain_modelstore`, which `brain caps` never touches. The table documents
# the ref a user would actually type (the auto-fetchable one), so these two
# are legitimate aliases, not documentation drift — same relationship as
# `brain_modelref::alias` maps a legacy short name to its canonical id, just
# in the other direction (a family placeholder to its recommended checkpoint).
CATALOG_ALIASES = {
    "brain/qwen": "Qwen/Qwen3-0.6B",
    "brain/lfm": "LiquidAI/LFM2.5-350M",
    "brain/z-image": "Tongyi-MAI/Z-Image-Turbo",
    "brain/yolo": "Ultralytics/YOLOv8",
}

ROW_RE = re.compile(r"^\|\s*\[`([^`]+)`\]\(([^)]+)\)")


def table_rows(readme: Path) -> list[tuple[str, str]]:
    text = readme.read_text()
    m = re.search(r"^## Model support\n(.*?)\n## ", text, re.S | re.M)
    if not m:
        print("FAIL: README.md has no '## Model support' section", file=sys.stderr)
        sys.exit(1)
    return [mm.groups() for line in m.group(1).splitlines() if (mm := ROW_RE.match(line))]


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

    rows = table_rows(args.repo / "README.md")
    table_ids = {r[0] for r in rows}
    problems: list[str] = []

    for cid in catalog_ids(args.brain):
        if cid in IGNORE:
            continue
        wanted = CATALOG_ALIASES.get(cid, cid)
        if wanted not in table_ids:
            problems.append(f"'{cid}' is in `brain caps` but '{wanted}' is not in README.md's Model support table")

    for model_id, link in rows:
        if not (args.repo / link).is_file():
            problems.append(f"'{model_id}' links to '{link}', which does not exist")

    if problems:
        print(f"FAIL: {len(problems)} problem(s):", file=sys.stderr)
        for prob in problems:
            print(f"  - {prob}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {len(table_ids)} table row(s), all links resolve, all `brain caps` ids documented")


if __name__ == "__main__":
    main()
