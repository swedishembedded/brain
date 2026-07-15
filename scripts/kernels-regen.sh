#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# kernels-regen.sh — regenerate the const block + ALL registry in
# crates/kernels/src/lib.rs from the contents of crates/kernels/wgsl/.
# This makes the registry mechanically derivable, so merge conflicts in
# lib.rs are resolved by union-ing wgsl/ and re-running this script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$REPO_ROOT/crates/kernels/src/lib.rs"
WGSL_DIR="$REPO_ROOT/crates/kernels/wgsl"

[ -f "$LIB" ] || { echo "missing $LIB" >&2; exit 1; }

# Preserve the doc header (everything before the first `/// \`wgsl/` line)
# and the tail (everything from `pub fn src` on, INCLUDING its /// doc block).
header_end="$(grep -n '^/// `wgsl/' "$LIB" | head -1 | cut -d: -f1)"
tail_start="$(grep -n '^pub fn src' "$LIB" | head -1 | cut -d: -f1)"
[ -n "$header_end" ] && [ -n "$tail_start" ] || { echo "unexpected lib.rs layout" >&2; exit 1; }
# Walk back over the contiguous doc-comment block above `pub fn src`.
while [ "$tail_start" -gt 1 ] && sed -n "$((tail_start - 1))p" "$LIB" | grep -q '^///'; do
  tail_start=$((tail_start - 1))
done

tmp="$(mktemp)"
head -n "$((header_end - 1))" "$LIB" > "$tmp"

# Const block, sorted by file stem (C-locale sort for stability).
for f in $(ls "$WGSL_DIR"/*.wgsl | LC_ALL=C sort); do
  stem="$(basename "$f" .wgsl)"
  upper="$(echo "$stem" | tr '[:lower:]' '[:upper:]')"
  printf '/// `wgsl/%s.wgsl`\npub const %s: &%s = include_str!("../wgsl/%s.wgsl");\n' \
    "$stem" "$upper" "str" "$stem" >> "$tmp"
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
