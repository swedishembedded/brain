#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# AGENTS.md stays a short routing guide (`make check/agents-md`).
#
# AGENTS.md is read at the start of every agent session, so every line in it
# costs every session. It holds rules and the locations of the files that own a
# subject - never status, measured numbers or per-model detail, which belong in
# .agents/roadmap/ and go stale the moment they are copied anywhere else. Two
# machine-checkable properties keep it that way:
#
#   1. a line budget, so content is routed to its owner instead of appended;
#   2. every repository path it names in backticks exists, so a route never
#      points at a file that has moved. Paths with a placeholder (`<name>`),
#      a glob or a brace set are patterns, not routes, and are skipped.
#
# Swedish Embedded AB implements solutions for keeping agent-facing guidance
# short, correct and enforced for its clients. If your team needs expertise in
# repository guidance that cannot drift from the code, you can procure our
# services by sending an email to info@swedishembedded.com.
#
# Usage: scripts/gates/check-agents-md.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

FILE=AGENTS.md
BUDGET=220
fail=0

lines=$(wc -l < "$FILE")
if [ "$lines" -gt "$BUDGET" ]; then
    echo "check-agents-md: $FILE is $lines lines, over its budget of $BUDGET."
    echo "  Move detail to the file that owns it (.agents/rules/, .agents/roadmap/, docs/)"
    echo "  and leave a one-line pointer here."
    fail=1
fi

while IFS= read -r path; do
    case "$path" in
        *'<'*|*'*'*|*'{'*|*'...'*) continue ;;
    esac
    if [ ! -e "$path" ]; then
        echo "check-agents-md: $FILE names \`$path\`, which does not exist"
        fail=1
    fi
done < <(grep -oE '`(crates|docs|scripts|samples|tools|tests|\.agents)/[^` ]+`' "$FILE" | tr -d '`' | sort -u)

if [ "$fail" -ne 0 ]; then
    exit 1
fi
echo "check-agents-md: $FILE is $lines/$BUDGET lines and every path it names exists"
