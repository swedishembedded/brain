#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# test-times.sh — rank every test binary by wall time.
#
# `cargo test` reports pass/fail but not *where the time went*, so a suite can
# drift from two minutes to an hour without anyone seeing which target did it.
# This runs each already-built test binary on its own and prints a ranked table,
# which is the measurement you need before deciding what to move out of the fast
# lane.
#
# Build first (`make build/release`), then:
#   scripts/gates/test-times.sh              # rank every binary
#   scripts/gates/test-times.sh --top 10     # just the worst offenders
#   scripts/gates/test-times.sh --budget 5   # exit non-zero if any binary exceeds 5s
#
# Deliberately runs binaries SEQUENTIALLY: brain's tests share one GPU, and
# timing them concurrently measures contention rather than cost.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PROFILE=${PROFILE:-release}
DEPS="target/$PROFILE/deps"
TOP=0
BUDGET=0
TIMEOUT=${TIMEOUT:-300}

while [ $# -gt 0 ]; do
  case "$1" in
    --top) TOP="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown flag $1"; exit 2 ;;
  esac
done

[ -d "$DEPS" ] || { echo "no $DEPS - run 'make build/release' first"; exit 2; }

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

for path in "$DEPS"/*; do
  b=$(basename "$path")
  # Test binaries are `<name>-<16 hex>`; skip .d/.rlib/.rmeta/.so artifacts.
  [[ "$b" =~ ^[a-z_0-9]+-[0-9a-f]{16}$ ]] || continue
  [ -x "$path" ] || continue
  n=$("$path" --list 2>/dev/null | grep -c ': test$')
  [ "${n:-0}" -gt 0 ] || continue
  s=$(date +%s.%N)
  timeout "$TIMEOUT" "$path" >/dev/null 2>&1
  rc=$?
  e=$(date +%s.%N)
  case $rc in
    0) status=ok ;;
    124) status=TIMEOUT ;;
    *) status=FAIL ;;
  esac
  printf "%8.1f %-7s %4d %s\n" "$(echo "$e-$s" | bc)" "$status" "$n" "${b%-*}" >>"$tmp"
done

echo
printf "%8s %-7s %4s %s\n" "seconds" "status" "n" "target"
printf -- "----------------------------------------------------------\n"
if [ "$TOP" -gt 0 ]; then
  sort -rn "$tmp" | head -n "$TOP"
else
  sort -rn "$tmp"
fi

total=$(awk '{s+=$1} END {printf "%.0f", s}' "$tmp")
count=$(wc -l <"$tmp")
echo
echo "$count test binaries, ${total}s sequential"

# Anything that times out or fails is worth surfacing even without --budget.
bad=$(awk '$2!="ok"' "$tmp" | wc -l)
[ "$bad" -gt 0 ] && echo "$bad binary(ies) did not finish cleanly:" && awk '$2!="ok"' "$tmp"

if [ "$BUDGET" -gt 0 ]; then
  over=$(awk -v b="$BUDGET" '$1>b' "$tmp")
  if [ -n "$over" ]; then
    echo
    echo "over the ${BUDGET}s per-target budget:"
    echo "$over"
    echo
    echo "Either make it faster, or move it to the slow lane with"
    echo "  #[ignore = \"slow: <why>\"]   (run via 'make test/slow')"
    exit 1
  fi
fi
exit 0
