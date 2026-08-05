#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# The clippy gate — a RATCHET, not a cliff.
#
# Why this exists: `cargo clippy` stops at the first deny-by-default lint and
# then reports NOTHING about anything scheduled after it, while still writing a
# pile of output. Grepping that output for ": warning:" therefore returns 0 for
# a run that linted almost nothing, and a warm target dir hides even the exit
# code (cargo replays diagnostics only for units it re-runs). That failure mode
# hid a 123-file backlog TWICE in one day — once from an agent, once from a
# careful human reading of the same output.
#
# So this script checks two separate things:
#   1. clippy EXITS 0. A non-zero exit means the lint pass aborted and every
#      count below it is meaningless. This is the check that was missing.
#   2. the warning count does not EXCEED the recorded baseline. New code must
#      not add warnings; the existing backlog is burned down by lowering the
#      baseline as it shrinks, which is what makes this adoptable today rather
#      than after someone fixes 190 lints in one sitting.
#
# Lower BASELINE whenever you clear warnings. Never raise it.
#
# Usage:
#   scripts/gates/clippy-gate.sh            # gate against BASELINE
#   scripts/gates/clippy-gate.sh --list     # print the current warnings and exit 0
#   BASELINE=n scripts/gates/clippy-gate.sh # override for a one-off check
set -uo pipefail

ROOT="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
cd "$ROOT"

# The number of clippy warnings the workspace currently carries. Every one is
# pre-existing; see task #24 and docs/imaging/plan.md. The largest group is
# doc-list indentation, which needs per-site judgment — an automated pass
# reattached a summary line to the wrong list item, which is a documentation
# defect rather than a lint fix.
#
# 207 -> 185 by fixing, not by suppressing: bfe00f3, 86bf582, and the yolo doc pass.
#
# LOWER THIS WHENEVER YOU CLEAR WARNINGS. Raising it is almost always wrong: it
# is how a real regression gets absorbed. It was raised exactly once, 179 -> 207,
# when this branch rebased onto 73 upstream commits that carry their own
# backlog. That is a different TREE, not worse code, and it was verified rather
# than assumed: of the 207, **zero** land on a line the branch added or changed
# (checked by intersecting each warning's file:line against the branch's own
# `git diff -U0 origin/main HEAD` hunks), and zero are in any of the thirteen
# crates the branch created. Anything short of that evidence means fix the
# warnings, not the number.
BASELINE="${BASELINE:-185}"

# Force a full re-lint: cargo replays diagnostics only for units it re-runs, so
# a warm target dir would otherwise report a small fraction of the real count.
find crates -name 'lib.rs' -o -name 'main.rs' | xargs touch

out="$(mktemp)"
trap 'rm -f "$out"' EXIT
cargo clippy --workspace --all-targets --message-format=short >"$out" 2>&1
rc=$?

if [ "$rc" -ne 0 ]; then
    echo "clippy-gate: FAIL — clippy exited $rc, so the lint pass ABORTED."
    echo "Everything scheduled after the offending crate went unlinted, and any"
    echo "warning count from this run is meaningless. Fix the error first:"
    echo
    grep -E '^error' "$out" | head -20
    exit 1
fi

if [ "${1:-}" = "--list" ]; then
    grep -E ': warning:' "$out"
    exit 0
fi

n=$(grep -cE ': warning:' "$out")
echo "clippy-gate: exit 0, $n warnings (baseline $BASELINE)"

if [ "$n" -gt "$BASELINE" ]; then
    echo
    echo "clippy-gate: FAIL — $n warnings exceeds the baseline of $BASELINE."
    echo "New code must not add clippy warnings. The additions are likely among:"
    echo
    grep -E ': warning:' "$out" | tail -"$((n - BASELINE))"
    exit 1
fi

if [ "$n" -lt "$BASELINE" ]; then
    echo "clippy-gate: $((BASELINE - n)) fewer than the baseline — lower BASELINE in $0 to $n."
fi
exit 0
