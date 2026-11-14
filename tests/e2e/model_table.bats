#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Drift guard for the model catalog (docs/reference/models/): see
# tests/e2e/model_table_check.py's module docs for exactly what this checks
# and (importantly) what it deliberately does NOT check.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary (build with: make build/debug, or set BRAIN_BIN)"
  export BRAIN
  PY="${EXAMPLES_PY:-python3}"
  export PY
}

@test "every \`brain caps\` model is named on a model page and the catalog's links resolve" {
  run "$PY" "$REPO/tests/e2e/model_table_check.py" --repo "$REPO" --brain "$BRAIN"
  if [ "$status" -ne 0 ]; then
    echo "$output" >&3
  fi
  [ "$status" -eq 0 ]
  [[ "$output" == "OK: "* ]]
}
