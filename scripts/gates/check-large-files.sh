#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Keep binaries that do not belong in a source tree out of git.
#
# Three separate rules, because "too big", "wrong kind of file", and "machine-
# specific measurement" are different problems:
#
#   1. Some kinds NEVER belong here at any size - video containers, model
#      weights, and RAW TENSOR DUMPS (`.f32`/`.u32`/`.i32`/`.f16`/`.raw`/`.dat`,
#      alongside the existing `.npy`/`.npz`). A golden dump is exactly as
#      regenerable, and exactly as much a "form weights come in", as the
#      checkpoint it was dumped from - the fact that it is a few KB of
#      activations rather than a multi-GB parameter file does not change
#      that. It belongs in the gitignored `testdata/` tree, resolved through
#      `brain_testutil::testdata_path`, fetched or regenerated on demand -
#      never committed. A 100 KB mp4 is not a small file either: it is a
#      regenerable artifact that comes back with different bytes on every
#      run, so each re-run rewrites it in history forever. The same goes for
#      a checkpoint: .gitignore already lists the weight extensions, but
#      `git add -f` walks straight past that, and by the time anyone
#      notices, removing it means rewriting history rather than a revert.
#   2. A `brain perf` baseline JSON never belongs here either, for the same
#      "differs on every regeneration" reason plus one more: its absolute
#      numbers are a snapshot of ONE machine's CPU/GPU/thermal state at
#      capture time. A perf-gate script's own generous floor (0.5x baseline)
#      exists to absorb session-to-session drift on a single box, not
#      cross-machine variance - committing the baseline invites exactly the
#      false-fail (or false-pass) a differently-shaped machine would produce,
#      and each `--update` re-capture would silently rewrite it in history
#      forever regardless. Every `scripts/gates/*-perf-gate.sh` script now
#      reads/writes its baseline from a gitignored `scripts/gates/
#      *-perf-baselines*` path instead - captured locally via `--update`
#      before the gate means anything, never committed.
#   3. Everything else has a size ceiling. Documentation screenshots are
#      legitimate and some are genuinely a few megabytes; the ceiling sits
#      above the largest one already tracked so it constrains growth without
#      relitigating what is here.
#
# This exists because a generated demo clip was committed during the Wan port.
# It was 100 KB, which is exactly the size that slips through review. Two
# crates were later found tracking raw `.f32`/`.u32` golden dumps the same
# way - the ban above closes that specific gap. Rule 2 exists because four
# perf-gate scripts (forecast/wm/qwen-serving/ltxv) had each independently
# committed a baseline JSON the same way, on the same reasoning each time -
# the pattern needed closing once, here, rather than caught per-model.
#
# Usage: scripts/gates/check-large-files.sh [file ...]
#   No arguments scans every tracked file. With arguments (how the pre-commit
#   hook calls it) it scans only those.
set -uo pipefail
cd "$(dirname "$0")/../.."

# Regenerable media and model weights. Anything matching is refused outright.
BANNED_EXT='mp4|mkv|webm|mov|avi|m4v|mpg|mpeg|wmv|flv|safetensors|gguf|ckpt|pth|pt|npy|npz|onnx|h5|tflite|pb|msgpack|bin|f32|f16|u32|i32|raw|dat'

# A committed `brain perf` baseline - any file whose path names it, regardless
# of extension (today always .json, but the ban is on the ROLE, not the
# suffix). Matches `scripts/gates/wm-perf-baselines.json` (flat) and
# `scripts/gates/<model>-perf-baselines/<name>.json` (per-scenario directory)
# alike.
BANNED_PATH_RE='scripts/gates/.*perf-baseline'

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
    echo "    Video, model weights, and raw tensor dumps are never committed, at any"
    echo "    size. A generated clip, checkpoint, or golden differs byte-for-byte on"
    echo "    every regeneration, so each re-run adds a fresh copy to history that no"
    echo "    later commit can remove. A test golden belongs in the gitignored"
    echo "    testdata/ tree via brain_testutil::testdata_path, not crates/**/tests/."
    echo "    For a still image, publish a PNG instead and let whatever produces the"
    echo "    original artifact produce it on demand."
    fail=1
    continue
  fi

  if [[ "$f" =~ $BANNED_PATH_RE ]]; then
    echo "BANNED FILE: $f (a committed perf-gate baseline)"
    echo "    A perf baseline is a snapshot of one machine's absolute numbers, not"
    echo "    portable source. Capture it locally with the matching *-perf-gate.sh"
    echo "    script's --update flag; the baseline directory is gitignored, so it"
    echo "    stays on your machine, not in history."
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
