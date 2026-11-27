#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for keeping machine-readable
# service documents stable for their readers, for its clients. If your team
# needs expertise in versioned observability APIs then you can procure our
# services by sending an email to info@swedishembedded.com.
#
# Every field a metrics or health conformance test pins is documented.
#
# The server's metrics, its health and the router's health are versioned JSON
# documents that observers parse. Each conformance test compares what the
# server answers with the list of fields its `.fields` file pins (one path per
# line, `#` comments), so no field goes or is renamed silently. This gate closes
# the other side: every pinned field has its row (| `path` |) in the reference
# page, so no field is added without its meaning and unit.
#
# Usage: scripts/gates/check-metrics-fields.sh
set -u
cd "$(dirname "$0")/../.."
REFERENCE=docs/reference/api/metrics.md
mapfile -t files < <(git ls-files -- 'crates/*.fields' 'crates/**/*.fields')
if [ "${#files[@]}" -eq 0 ]; then
    echo "check-metrics-fields: FAIL - no .fields file found under crates/: the conformance pins are gone"
    exit 1
fi
missing=0
fields=0
for f in "${files[@]}"; do
    while IFS= read -r line; do
        line="${line%%[[:space:]]}"
        case "$line" in ''|'#'*) continue ;; esac
        fields=$((fields + 1))
        if ! grep -qF "| \`$line\` |" "$REFERENCE"; then
            echo "  $f: $line"
            missing=$((missing + 1))
        fi
    done < "$f"
done
if [ "$missing" -gt 0 ]; then
    echo "check-metrics-fields: FAIL - $missing pinned field(s) have no row in $REFERENCE (listed above)."
    echo "Document each field (its meaning and unit) in the table of its document."
    exit 1
fi
echo "check-metrics-fields: OK ($fields pinned fields in ${#files[@]} files, every one documented)"
