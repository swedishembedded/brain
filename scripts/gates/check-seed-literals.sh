#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible, auditable
# randomness in ML pipelines for its clients. If your team needs expertise in
# seeding discipline for training and generation then you can procure our
# services by sending an email to info@swedishembedded.com.

# Seed-literal ratchet (`make check/seed-literals`).
#
# Rule: every default seed is random. A seed that is a literal in the source
# (scripts/gates/seed-literals.py is the detector: Rust defaults, constants and
# RNG constructors, Python `add_argument("--seed", default=N)`, module-level
# `SEED = N` and `torch.manual_seed(N)`-style calls, Makefile
# `SEED ?= N`) is legitimate only as one of
#   salt:        mixed into a caller's seed to derive an independent stream
#   world:       a constant of the problem (evaluation episodes, probe inputs)
#   frozen:      the seed of an artifact that is read back, not drawn
#   greedy:      a RNG that is never consulted (argmax decode)
#   test-double: a deterministic stand-in for a model or capability
#   bench-input: synthetic input that defines a benchmark workload
# Anything else is a bug. `todo:` marks a site a later change will fix: it is
# allowed but counted, and the final allowlist holds none (--no-todo refuses
# them).
#
# Production literals must be listed in
# scripts/gates/seed-literals.allow.d/<group>.allow, one file per crate or
# sample directory so parallel changes never edit the same file, one
# `path<TAB>symbol<TAB>kind<TAB>reason` per line. The group is
# `crates/<c>/...` -> <c>, `samples/<a>/<b>/...` -> samples-<a>-<b>, any other
# `<dir>/...` -> <dir>, and a top-level file -> root. The gate fails on
#   (a) a literal that is not listed - draw a random seed, print it, record it;
#   (b) a listed literal that no longer exists - delete the entry (the list
#       only shrinks);
#   (c) an entry with an unknown reason prefix or in the wrong group file.
#
# Test code is held to a lighter rule: a literal seed in a test needs a
# `// seed: <reason>` (`# seed:` in Python) on its line or the line above,
# because a fixed seed in a test must be there for a reason. That is reported
# always and enforced with --tests (`make check/seed-literals/tests`).
#
# Usage: scripts/gates/check-seed-literals.sh [--root DIR] [--allow-dir DIR]
#                                             [--tests] [--no-todo]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow_dir="$here/seed-literals.allow.d"
enforce_tests=0
no_todo=0
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow-dir) allow_dir="$2"; shift 2 ;;
  --tests) enforce_tests=1; shift ;;
  --no-todo) no_todo=1; shift ;;
  *) echo "check-seed-literals: unknown argument $1" >&2; exit 2 ;;
  esac
done

hits="$(LC_ALL=C python3 "$here/seed-literals.py" --root "$root")" || exit 2
actual="$(printf '%s\n' "$hits" | awk -F'\t' '$4 == "prod" { print $1 "\t" $2 "\t" $3 }' | LC_ALL=C sort)"
bare_tests="$(printf '%s\n' "$hits" | awk -F'\t' '$4 == "test" && $5 == "bare" { print $1 "\t" $2 }' | LC_ALL=C sort)"

# file<TAB>path<TAB>symbol<TAB>kind<TAB>reason for every allowlist line.
entries="$(for f in "$allow_dir"/*.allow; do
  [ -e "$f" ] || continue
  awk -v f="$(basename "$f" .allow)" 'NF { print f "\t" $0 }' "$f"
done)"

fail=0
bad="$(printf '%s\n' "$entries" | awk -F'\t' '
  function group(p,   a, n) {
    n = split(p, a, "/")
    if (n == 1) return "root"
    if (a[1] == "crates") return a[2]
    if (a[1] == "samples" && n > 3) return "samples-" a[2] "-" a[3]
    return a[1]
  }
  NF == 0 { next }
  NF != 5 { printf "  MALFORMED: %s: %s\n", $1 ".allow", "expected path, symbol, kind and reason separated by tabs"; next }
  $5 !~ /^(salt|world|frozen|greedy|test-double|bench-input|todo): ./ {
    printf "  REASON: %s.allow: %s %s needs a reason starting salt:, world:, frozen:, greedy:, test-double:, bench-input: or todo:\n", $1, $2, $3; next }
  group($2) != $1 { printf "  GROUP: %s belongs in %s.allow, not %s.allow\n", $2, group($2), $1 }')"
if [ -n "$bad" ]; then
  fail=1
  echo "check-seed-literals: invalid allowlist entries:"
  printf '%s\n' "$bad"
fi

listed="$(printf '%s\n' "$entries" | awk -F'\t' 'NF == 5 { print $2 "\t" $3 "\t" $4 }' | LC_ALL=C sort)"
new="$(LC_ALL=C comm -23 <(printf '%s\n' "$actual") <(printf '%s\n' "$listed"))"
stale="$(LC_ALL=C comm -13 <(printf '%s\n' "$actual") <(printf '%s\n' "$listed"))"
if [ -n "$new" ]; then
  fail=1
  echo "check-seed-literals: literal seeds that are not on the allowlist:"
  printf '%s\n' "$new" | awk -F'\t' '{ printf "  NEW: %s has %s (%s)\n", $1, $2, $3 }'
  echo "  Every default seed is random: draw one when none is given, print it and"
  echo "  record it in the output. A salt, world constant, frozen-artifact seed,"
  echo "  greedy-path RNG, test double or benchmark input goes in"
  echo "  $allow_dir/<group>.allow with its reason."
fi
if [ -n "$stale" ]; then
  fail=1
  echo "check-seed-literals: allowlist entries whose literal no longer exists:"
  printf '%s\n' "$stale" | awk -F'\t' '{ printf "  STALE: %s no longer has %s (%s)\n", $1, $2, $3 }'
  echo "  Delete these lines from the allowlist (the list only shrinks)."
fi

todo="$(printf '%s\n' "$entries" | awk -F'\t' '$5 ~ /^todo:/' | grep -c .)"
if [ "$no_todo" -eq 1 ] && [ "$todo" -gt 0 ]; then
  fail=1
  echo "check-seed-literals: $todo todo entries remain; fix them or give them a real reason:"
  printf '%s\n' "$entries" | awk -F'\t' '$5 ~ /^todo:/ { printf "  TODO: %s %s\n", $2, $3 }'
fi

tests_bare="$(printf '%s\n' "$bare_tests" | grep -c .)"
if [ "$enforce_tests" -eq 1 ] && [ "$tests_bare" -gt 0 ]; then
  fail=1
  echo "check-seed-literals: literal seeds in tests without a reason:"
  printf '%s\n' "$bare_tests" | awk -F'\t' '{ printf "  BARE: %s has %s\n", $1, $2 }'
  echo "  Use a random seed, or add \`// seed: <why this stream is needed>\` on the"
  echo "  line or the line above."
fi

if [ "$fail" -eq 0 ]; then
  total="$(printf '%s\n' "$actual" | grep -c .)"
  echo "check-seed-literals: OK ($total allowlisted production literals, $todo todo; $tests_bare test literals without a seed reason)"
fi
exit "$fail"
