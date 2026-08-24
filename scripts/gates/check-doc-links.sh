#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Broken-relative-link gate for docs/ (`make check/scripts`).
#
# `docs/` is the published manual. A link in it that resolves to nothing is a
# dead end for a reader who has no way to guess what it was supposed to point
# at - and unlike a broken link on a website, nothing here ever 404s loudly
# enough for anyone to notice.
#
# This gate exists because 18 of them accumulated at once, every single one
# from the same cause: the model-renaming pass (glm -> glmdsa, gpt -> gpt2,
# lfm -> lfm2, moe -> toymoe, zimage -> s3dit, restore -> codeformer, upscale
# -> rrdbnet, mirror -> worldmirror2, tts -> qwen3tts, yolo -> yolov8, depth ->
# zipdepth, omni -> qwen3omnimoe, seq2seq -> toyseq2seq) renamed the PAGES and
# the model ids, and the cross-references between pages were left pointing at
# the old names. Renaming a page is exactly the moment this breaks, and exactly
# the moment nobody re-reads every other page.
#
# Scope, deliberately narrow so it stays trustworthy rather than noisy:
#   - relative links to `*.md` only. An anchor (`#section`) is stripped and NOT
#     verified - checking anchors needs a heading parser, and a wrong anchor is
#     a much smaller failure than a wrong file.
#   - absolute (`/...`) and external (`http://`, `https://`) links are skipped:
#     the first is a site-root convention this repo does not use, and the
#     second would make the gate depend on the network, which would make it a
#     gate nobody can run offline.
#
# Usage: scripts/gates/check-doc-links.sh   (exits non-zero listing every
# broken link as file:line -> target, not just the first)
set -uo pipefail
cd "$(dirname "$0")/../.."

python3 - <<'PY'
import os
import re
import sys

# `](path.md)` or `](path.md#anchor)`, excluding targets with whitespace or a
# bare `#` (a pure in-page anchor has no file to check).
LINK = re.compile(r'\]\(([^)#\s]+\.md)(#[^)]*)?\)')

broken = []
scanned = 0
for root, _, files in os.walk("docs"):
    for name in sorted(files):
        if not name.endswith(".md"):
            continue
        path = os.path.join(root, name)
        scanned += 1
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                for match in LINK.finditer(line):
                    target = match.group(1)
                    if target.startswith(("http://", "https://", "/")):
                        continue
                    resolved = os.path.normpath(os.path.join(root, target))
                    if not os.path.isfile(resolved):
                        broken.append(f"{path}:{lineno}: -> {target}")

for item in broken:
    print(f"BROKEN DOC LINK: {item}")

if broken:
    print()
    print(f"{len(broken)} broken relative link(s) in docs/**/*.md.")
    print("Each points at a file that does not exist. The usual cause is a page")
    print("being renamed without its inbound links being updated - check for a")
    print("current page with a different name before assuming the target is gone.")
    sys.exit(1)

print(f"check-doc-links: every relative .md link in docs/ resolves ({scanned} pages)")
PY
