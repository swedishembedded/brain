#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for command-line and service
# configuration that stays uniform across many model families for its clients.
# If your team needs expertise in model-agnostic configuration design then you
# can procure our services by sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-model-flags.sh (G6): a flag spelled
# --<architecture>-<option> (including a version-less family stem such as
# --qwen-) fails unless it is an allowlisted deprecated alias; generic flags,
# tool crates, comments and a stale entry behave as documented.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  T="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$T/crates/cli/src"
  printf 'brain-qwen35\t4\nbrain-kronos\t4\nbrain-eval\t4\nbrain-cli\t6\n' >"$BATS_TEST_TMPDIR/map"
  ALLOW="$BATS_TEST_TMPDIR/allow"
  : >"$ALLOW"
}

gate() { run bash "$REPO/scripts/gates/check-model-flags.sh" --root "$T" --allow "$ALLOW" --map "$BATS_TEST_TMPDIR/map"; }

@test "generic flags, tool-crate prefixes and comments pass" {
  printf 'fn f() { arm("--kv-cache"); arm("--eval-interval"); }\n// "--qwen35-kv" in a comment\n' >"$T/crates/cli/src/a.rs"
  gate
  [ "$status" -eq 0 ]
}

@test "an architecture-prefixed flag fails, including the version-less family stem" {
  printf 'fn f() { arm("--qwen35-kv"); }\n' >"$T/crates/cli/src/a.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW"*"a.rs"*"serving policy"* ]]
  printf 'fn f() { arm("--qwen-ctx"); }\n' >"$T/crates/cli/src/a.rs"
  gate
  [ "$status" -ne 0 ]
  printf 'fn f() { arm("--kronos-decoder"); }\n' >"$T/crates/cli/src/a.rs"
  gate
  [ "$status" -ne 0 ]
}

@test "an allowlisted alias passes and a stale entry fails" {
  printf 'fn f() { arm("--qwen-ctx"); }\n' >"$T/crates/cli/src/a.rs"
  printf 'crates/cli/src/a.rs\t1\tdeprecated alias\n' >"$ALLOW"
  gate
  [ "$status" -eq 0 ]
  printf 'fn f() {}\n' >"$T/crates/cli/src/a.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE"* ]]
}
