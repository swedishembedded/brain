#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Regression guard for `make perf/compare` / `make bench/compare`'s file
# enumeration. `results/*.json` filenames can legitimately contain spaces,
# brackets and plus signs - crates/perf/src/schema.rs's `default_path()` builds
# the name from the resolved device set, e.g.
# `perf-serve-fake-gpu[0] + npu[0] + cpu[22 core(s)]-1234.json`. An unquoted
# `ls results/*.json` assigned to a shell var and then word-split (the old
# recipe) drops or mis-globs a filename like that; this test seeds one and
# checks it is picked up like any other artifact, not silently dropped and not
# an error.
#
# Run: make test/e2e/results-hygiene

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no release brain binary (build with: make build/release)"
  export BRAIN

  RESULTS="$REPO/results"
  export RESULTS
  TMP_BACKUP="$(mktemp -d)"
  export TMP_BACKUP
  # Move any real artifacts aside so the assertions below are exact, then
  # restore them in teardown - results/ holds real, uncommitted run output
  # that this test must not disturb.
  if [ -d "$RESULTS" ]; then
    mv "$RESULTS" "$TMP_BACKUP/results"
  fi
  mkdir -p "$RESULTS"
}

teardown() {
  rm -rf "$RESULTS"
  if [ -d "$TMP_BACKUP/results" ]; then
    mv "$TMP_BACKUP/results" "$RESULTS"
  fi
  rm -rf "$TMP_BACKUP"
}

# A minimal-but-valid brain.perf/1 artifact - every field report::load() reads
# falls back via unwrap_or when absent, so schema/scenario/target/env/valid is
# the whole contract.
fixture() {
  cat >"$1" <<JSON
{"schema":"brain.perf/1","scenario":"$2","valid":true,"target":{"model":"m","artifact_unit":"token"},"env":{"device":"cpu"}}
JSON
}

@test "make perf/compare picks up a results/*.json filename with spaces, brackets and a plus sign" {
  fixture "$RESULTS/perf-latency-cpu-1.json" latency
  fixture "$RESULTS/perf-throughput-cpu-2.json" throughput
  fixture "$RESULTS/perf-a b[0]+c-3.json" serve

  run make -C "$REPO" perf/compare BRAIN="$BRAIN"
  [ "$status" -eq 0 ]
  [[ "$output" != *"no such file"* ]]
  [[ "$output" == *"perf-latency-cpu-1"* ]]
  [[ "$output" == *"perf-throughput-cpu"* ]]
  [[ "$output" == *"perf-a b[0]+c-3"* ]]
}
