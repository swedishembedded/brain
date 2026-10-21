#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
# Swedish Embedded AB implements solutions for loading HuggingFace model releases as published for its clients. If
# your team needs expertise in keeping a model zoo loadable from upstream layouts then you can procure our services
# by sending an email to info@swedishembedded.com.

# HuggingFace support cannot regress silently (`make check/hf-fixtures`).
#
# An architecture that declares HuggingFace names in crates/arch (`hf: &[...]`) tells `brain pull` it can serve a
# repo of that class. That claim is only worth something while a test keeps loading a tiny HF-layout fixture (random
# weights, the release's own tensor names and shard layout) for it. This gate requires, for every architecture row
# with `hf:` names, either
#
#   * a test that says it covers the architecture with a marker comment `hf-fixture: <id>[, <id>...]` in a file under
#     crates/<crate>/tests/ or crates/<crate>/src/ (the marker sits in the test that plans and loads the fixture), or
#   * an entry in scripts/gates/hf-fixtures.allow (`<id> <reason>`): an architecture whose HF loader does not exist
#     yet or whose fixture is still to be written. The list only shrinks: an entry whose architecture now has a
#     marker, or no longer declares HF names, fails the gate.
#
# Usage: scripts/gates/check-hf-fixtures.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

declared=$(python3 -I - <<'PY'
import re
s = open("crates/arch/src/lib.rs").read()
body = s[s.index("pub const ARCHS"):s.index("#[cfg(test)]")]
for part in re.split(r"\barch!\(", body)[1:]:
    ident = re.match(r'\s*"([a-z0-9]+)"', part)
    if ident and re.search(r"hf:\s*&\[\s*\"", part):
        print(ident.group(1))
PY
)

markers=$(grep -rhoE 'hf-fixture:[ a-z0-9,]+' crates --include='*.rs' | sed 's/hf-fixture://' | tr ',' '\n' | tr -d ' ' | grep -v '^$' | sort -u)
allowed=$(grep -vE '^\s*(#|$)' scripts/gates/hf-fixtures.allow | awk '{print $1}' | sort -u)

fail=0
for id in $declared; do
  covered=0; excused=0
  grep -qx "$id" <<<"$markers" && covered=1
  grep -qx "$id" <<<"$allowed" && excused=1
  if [ $covered = 0 ] && [ $excused = 0 ]; then
    echo "check/hf-fixtures: '$id' declares HuggingFace names but no test carries 'hf-fixture: $id' and it is not in scripts/gates/hf-fixtures.allow" >&2
    fail=1
  fi
  if [ $covered = 1 ] && [ $excused = 1 ]; then
    echo "check/hf-fixtures: '$id' has a fixture marker; remove it from scripts/gates/hf-fixtures.allow" >&2
    fail=1
  fi
done
for id in $allowed; do
  grep -qx "$id" <<<"$declared" || { echo "check/hf-fixtures: allowlist entry '$id' no longer declares HuggingFace names; remove it" >&2; fail=1; }
done
for id in $markers; do
  grep -qx "$id" <<<"$declared" || { echo "check/hf-fixtures: a test claims 'hf-fixture: $id' but that architecture declares no HuggingFace names" >&2; fail=1; }
done
[ $fail = 0 ] && echo "check/hf-fixtures: ok ($(wc -w <<<"$declared") architectures declare HuggingFace names)"
exit $fail
