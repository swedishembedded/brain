#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Linear-history gate (`make check/scripts`).
#
# AGENTS.md/CLAUDE.md require the working branch's history to stay linear:
# commits are self-contained and rebased onto, never merged into, main. A
# merge commit means two branches were combined with `git merge` instead of
# `git rebase`, which is exactly the shape this gate exists to catch before it
# reaches a remote (see scripts/hooks/pre-push, which runs this against the
# full tree being pushed).
#
# Usage: scripts/gates/check-linear-history.sh [<ref>]   (default: HEAD)
set -u
cd "$(dirname "$0")/../.."

REF="${1:-HEAD}"

merges=$(git rev-list --merges "$REF" 2>&1)
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "check-linear-history: 'git rev-list --merges $REF' failed:"
  echo "$merges"
  exit 1
fi

if [ -n "$merges" ]; then
  count=$(printf '%s\n' "$merges" | wc -l)
  echo "check-linear-history: $count merge commit(s) found in $REF's history:"
  git log --merges --format='  %h %ad %s' --date=short "$REF" 2>/dev/null
  echo
  echo "check-linear-history: merge commits are banned. Rebase instead of merging."
  exit 1
fi

echo "check-linear-history: OK ($REF is linear)"
