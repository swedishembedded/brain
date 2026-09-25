#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# kernels-regen.sh — regenerate the const block + ALL registry in
# crates/kernels/src/lib.rs from the contents of crates/kernels/wgsl/.
# This makes the registry mechanically derivable, so merge conflicts in
# lib.rs are resolved by union-ing wgsl/ and re-running this script.
set -euo pipefail

# This script lives at scripts/build/kernels-regen.sh -- two levels below the
# repo root, not one. Same class of bug commit 96dc6b4 fixed in
# scripts/gates/*.sh (each `cd`'d one directory short of the repo root and
# every path built from REPO_ROOT silently resolved under scripts/ instead).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIB="$REPO_ROOT/crates/kernels/src/lib.rs"
WGSL_DIR="$REPO_ROOT/crates/kernels/wgsl"

[ -f "$LIB" ] || { echo "missing $LIB" >&2; exit 1; }

# Preserve the doc header (everything before the first `/// \`wgsl/` line)
# and the tail (everything from `pub fn src` on, INCLUDING its /// doc block).
#
# `grep -m1` rather than `grep | head -1`: under `set -o pipefail` (line 6), head
# closing the pipe SIGPIPEs grep, the pipeline exits 141, and `set -e` kills the
# script right here — after `header_end` has already been assigned, so the failure
# looks like nothing happening rather than like an error. It is a race on grep
# outrunning the pipe buffer, so it worked by luck for ~200 kernels and then
# started aborting the regen; `make kernels-regen` does surface the 141, but a
# caller that pipes the output (`| tail -1`) sees only silence and an unchanged
# registry. -m1 makes grep stop on its own, so nothing is ever SIGPIPEd.
header_end="$(grep -n -m1 '^/// `wgsl/' "$LIB" | cut -d: -f1)"
tail_start="$(grep -n -m1 '^pub fn src' "$LIB" | cut -d: -f1)"
[ -n "$header_end" ] && [ -n "$tail_start" ] || { echo "unexpected lib.rs layout" >&2; exit 1; }
# Walk back over the contiguous doc-comment block above `pub fn src`.
while [ "$tail_start" -gt 1 ] && sed -n "$((tail_start - 1))p" "$LIB" | grep -q '^///'; do
  tail_start=$((tail_start - 1))
done

tmp="$(mktemp)"
head -n "$((header_end - 1))" "$LIB" > "$tmp"

# Const block, sorted by file stem (C-locale sort for stability).
#
# A kernel that needs shared device code names it in its header as
# `// @import <name>` (one per line, in order); the library lives at
# `wgsl/lib/<name>.wgsl` and is concatenated IN FRONT of the kernel at compile
# time with `concat!`, so the registered source is still one self-contained
# WGSL module and every backend (wgpu, the CPU JIT, native Vulkan, the CUDA
# translation) sees exactly what it always did. Libraries hold functions and
# structs only - no entry point, no bindings - so they are not kernels and are
# not registered on their own.
for f in $(ls "$WGSL_DIR"/*.wgsl | LC_ALL=C sort); do
  stem="$(basename "$f" .wgsl)"
  upper="$(echo "$stem" | tr '[:lower:]' '[:upper:]')"
  imports="$(sed -n 's|^// @import \([a-z0-9_]*\)$|\1|p' "$f")"
  if [ -z "$imports" ]; then
    printf '/// `wgsl/%s.wgsl`\npub const %s: &%s = include_str!("../wgsl/%s.wgsl");\n' \
      "$stem" "$upper" "str" "$stem" >> "$tmp"
  else
    parts=""
    for lib in $imports; do
      [ -f "$WGSL_DIR/lib/$lib.wgsl" ] || { echo "$stem imports missing wgsl/lib/$lib.wgsl" >&2; exit 1; }
      parts="$parts    include_str!(\"../wgsl/lib/$lib.wgsl\"),\n"
    done
    printf '/// `wgsl/%s.wgsl`, after `wgsl/lib/{%s}.wgsl`\npub const %s: &%s = concat!(\n%b    include_str!("../wgsl/%s.wgsl"),\n);\n' \
      "$stem" "$(echo $imports | tr ' ' ',')" "$upper" "str" "$parts" "$stem" >> "$tmp"
  fi
done

# ALL registry.
printf '\n/// Every kernel as `(name, source)`, name = file stem.\npub const ALL: &[(&%s, &%s)] = &[\n' "str" "str" >> "$tmp"
for f in $(ls "$WGSL_DIR"/*.wgsl | LC_ALL=C sort); do
  stem="$(basename "$f" .wgsl)"
  upper="$(echo "$stem" | tr '[:lower:]' '[:upper:]')"
  printf '    ("%s", %s),\n' "$stem" "$upper" >> "$tmp"
done
printf '];\n\n' >> "$tmp"

tail -n "+$tail_start" "$LIB" >> "$tmp"
mv "$tmp" "$LIB"

n="$(ls "$WGSL_DIR"/*.wgsl | wc -l)"
echo "kernels-regen: $n kernels registered in crates/kernels/src/lib.rs"
