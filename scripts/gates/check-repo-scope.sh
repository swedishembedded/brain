#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Repository scope: a tracked file names another project only if brain depends
# on it. brain is the model engine; it depends on no agent runtime, learning
# application or orchestration service, so none of those is named anywhere in
# the tree. A caller is described generically instead ("an agent runtime", "an
# application", "an orchestrator"). AGENTS.md "Repository scope" states the
# rule and what is in and out of scope.
#
# The names are matched as whole words, case-insensitively. "whale" is also an
# ordinary English word used in the image-generation examples ("a whale
# submarine", "a submarine shaped like a whale", "whale.png"), so a whale hit
# on a line that also contains "submarine" or "whale.png" is not a project
# mention and is filtered out.
#
# CHANGELOG.md is history (generated from past commit subjects) and exempt, as
# is this script, which has to spell the names to match them. So is
# .gitignore: an ignore pattern for a tool's state directory is a filter, not
# a description of that tool.
#
# Usage: scripts/gates/check-repo-scope.sh [file ...]
#   With no arguments, scans every tracked file. With arguments (how the
#   pre-commit hook calls it), scans only the files given.
set -uo pipefail
cd "$(dirname "$0")/../.."

SELF="scripts/gates/check-repo-scope.sh"
NAME_PATTERN='\b(sven|splinter)\b'
WHALE_PATTERN='\bwhale\b'
WHALE_WORD_PATTERN='submarine|whale\.png'

files=()
if [ "$#" -gt 0 ]; then
  candidates=("$@")
else
  mapfile -d '' -t candidates < <(git ls-files -z)
fi
for f in "${candidates[@]}"; do
  case "$f" in
  CHANGELOG.md | .gitignore | "$SELF") continue ;;
  esac
  # A tracked path deleted in the working tree has nothing left to scan.
  [ -f "$f" ] && files+=("$f")
done

[ "${#files[@]}" -eq 0 ] && { [ "$#" -eq 0 ] && echo "check-repo-scope: OK"; exit 0; }

# -I skips binary files; -H keeps the file name even when one file is given.
# xargs splits long file lists; grep exits 1 on "no match", which is success.
name_hits=$(printf '%s\0' "${files[@]}" | xargs -0 grep -IHniP "$NAME_PATTERN" 2>/dev/null)
whale_hits=$(printf '%s\0' "${files[@]}" | xargs -0 grep -IHniP "$WHALE_PATTERN" 2>/dev/null |
  grep -viP "$WHALE_WORD_PATTERN")

hits=$(printf '%s\n%s\n' "$name_hits" "$whale_hits" | sed '/^$/d' | sort -t: -k1,1 -k2,2n)

[ -z "$hits" ] && { [ "$#" -eq 0 ] && echo "check-repo-scope: OK"; exit 0; }

echo "check-repo-scope: tracked file names a project brain does not depend on:"
echo "$hits" | cut -c1-240 | sed 's/^/  /'
cat <<'EOF'

brain names another project only if it depends on it. Describe the caller
generically instead: "an agent runtime", "an application", "an orchestrator"
(e.g. "an agent's request", "an orchestrator's execution cache",
"graph-executing callers"). See AGENTS.md "Repository scope".
EOF
exit 1
