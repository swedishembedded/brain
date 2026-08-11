#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Pre-commit / CI gate: code never cites a docs/ or .agents/ file path.
#
# docs/ is user-facing product documentation and .agents/ is contributor
# rules/roadmaps — both move and get rewritten over time. A comment or string
# in crates/, scripts/, tools/, or examples/ that names one of their paths
# (`docs/foo.md`, `.agents/rules/bar.md`, a `#N` lesson number, a `§` section)
# is a cross-reference nothing enforces: it silently goes stale the next time
# that doc is edited, renamed, or deleted, and the fix cost lands on whoever
# reads the dangling citation, not whoever moved the file. Every one of these
# found during the docs/ restructure had already rotted this way — some cited
# specs that had been deleted outright.
#
# The fix is not "keep the citation up to date" (that's the file-path
# staleness problem all over again), it's "don't cite a doc path from code at
# all" — state the fact or reasoning inline instead.
#
# This gate is deliberately narrow:
#   - it does not stop docs/ or .agents/ files from linking to each other
#     (that's normal), and it does not touch AGENTS.md/README.md (whose whole
#     job is routing to those trees);
#   - it exempts scripts/gates/ and scripts/hooks/: a documentation-structure
#     gate (like check-env-docs.sh, or this file) necessarily NAMES the real
#     doc paths it validates against, or describes them in its own header —
#     that is a functional dependency the gate fails loudly on if it goes
#     stale (the check breaks, visibly), not a decorative citation that goes
#     stale silently. That's the actual distinction this gate exists to
#     enforce, so this small tree is the one place doing the opposite is
#     correct.
#
# Usage:
#   scripts/gates/check-no-doc-citations.sh              # scan the whole tree
#   scripts/gates/check-no-doc-citations.sh <file> ...    # scan only these
#                                                          # (pre-commit passes
#                                                          # the staged files)
set -u
cd "$(dirname "$0")/../.."

PATTERN='\.agents/|docs/[A-Za-z0-9_./-]+\.md'
EXEMPT_RE='^scripts/(gates|hooks)/'

if [ "$#" -gt 0 ]; then
  files=()
  for f in "$@"; do
    case "$f" in
      scripts/gates/*|scripts/hooks/*) continue ;;
      crates/*|scripts/*|tools/*|examples/*|brain-py/*) files+=("$f") ;;
    esac
  done
  [ "${#files[@]}" -eq 0 ] && exit 0
  hits=$(grep -nE "$PATTERN" "${files[@]}" 2>/dev/null)
else
  hits=$(grep -rnE "$PATTERN" crates scripts tools examples brain-py 2>/dev/null | grep -vE "^$EXEMPT_RE")
fi

if [ -n "$hits" ]; then
  echo "CODE CITES A docs/ OR .agents/ FILE PATH — not allowed:"
  echo "$hits"
  echo
  echo "Remove the file-path/section/lesson-number citation and state the"
  echo "fact or reasoning inline instead. See scripts/gates/check-no-doc-citations.sh"
  echo "for why."
  exit 1
fi
echo "check-no-doc-citations: clean"
