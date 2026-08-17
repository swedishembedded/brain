#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Keep binaries that do not belong in a source tree out of git.
#
# Two separate rules, because "too big" and "wrong kind of file" are different
# problems:
#
#   1. Some kinds NEVER belong here at any size - video containers and model
#      weights. A 100 KB mp4 is not a small file, it is a regenerable artifact
#      that will be regenerated with different bytes on every run, so each
#      re-run rewrites it in history forever. The same goes for a checkpoint:
#      .gitignore already lists the weight extensions, but `git add -f` walks
#      straight past that, and by the time anyone notices, removing it means
#      rewriting history rather than a revert.
#   2. Everything else has a size ceiling. Documentation screenshots are
#      legitimate and some are genuinely a few megabytes; the ceiling sits
#      above the largest one already tracked so it constrains growth without
#      relitigating what is here.
#
# This exists because a generated demo clip was committed during the Wan port.
# It was 100 KB, which is exactly the size that slips through review.
#
# Usage: scripts/gates/check-large-files.sh [file ...]
#   No arguments scans every tracked file. With arguments (how the pre-commit
#   hook calls it) it scans only those.
set -uo pipefail
cd "$(dirname "$0")/../.."

# Regenerable media and model weights. Anything matching is refused outright.
BANNED_EXT='mp4|mkv|webm|mov|avi|m4v|mpg|mpeg|wmv|flv|safetensors|gguf|ckpt|pth|pt|npy|npz|onnx|h5|tflite|pb|msgpack|bin'

# Ceiling for everything else. The largest tracked file today is a ~4.9 MB
# quickstart screenshot, so this leaves headroom for a comparable one without
# permitting a step change.
MAX_BYTES=$((6 * 1024 * 1024))

if [ "$#" -gt 0 ]; then files=("$@"); else mapfile -t files < <(git ls-files); fi
fail=0

for f in "${files[@]}"; do
  [ -f "$f" ] || continue

  if [[ "$f" =~ \.(${BANNED_EXT})$ ]]; then
    echo "BANNED FILE TYPE: $f"
    echo "    Video and model weights are never committed, at any size. A generated"
    echo "    clip or checkpoint differs byte-for-byte on every regeneration, so each"
    echo "    re-run adds a fresh copy to history that no later commit can remove."
    echo "    Publish a still frame (PNG) instead, gitignore the artifact, and let"
    echo "    whatever produces it produce it."
    fail=1
    continue
  fi

  sz=$(wc -c <"$f")
  if [ "$sz" -gt "$MAX_BYTES" ]; then
    echo "FILE TOO LARGE: $f ($((sz / 1024)) KiB, ceiling $((MAX_BYTES / 1024)) KiB)"
    echo "    If this is a documentation image, compress it. If it is test data,"
    echo "    it belongs in the fixture tree that make fetch/testdata populates,"
    echo "    not in the repository."
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo
  echo "check-large-files: refusing the above. Override for a genuinely exceptional"
  echo "case is a deliberate change to this script, not a flag - so that the"
  echo "exception gets read by someone."
  exit 1
fi

[ "$#" -eq 0 ] && echo "check-large-files: OK ($((${#files[@]})) tracked files)"
exit 0
