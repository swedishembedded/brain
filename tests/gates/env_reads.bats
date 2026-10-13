#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Spec for scripts/gates/check-env-reads.sh, the env-read ratchet: a read that
# is not on the allowlist fails (use a CLI flag instead), an allowlist entry
# whose read is gone fails (the list only shrinks), an exact match passes.
# Each case runs the gate against a throwaway tree, never the real one.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  TREE="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$TREE/crates/a/src"
  printf 'fn f() { let _ = std::env::var("BRAIN_OLD"); }\n' >"$TREE/crates/a/src/lib.rs"
  printf 'crates/a/src/lib.rs\tBRAIN_OLD\n' >"$BATS_TEST_TMPDIR/allow"
}

gate() {
  run bash "$REPO/scripts/gates/check-env-reads.sh" --root "$TREE" --allow "$BATS_TEST_TMPDIR/allow"
}

@test "an allowlisted read passes" {
  gate
  [ "$status" -eq 0 ]
}

@test "a new read fails and points at CLI flags" {
  printf 'fn g() { let _ = std::env::var_os("BRAIN_NEW"); }\n' >>"$TREE/crates/a/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW READ"*"BRAIN_NEW"* ]]
  [[ "$output" == *"flag"* ]]
}

@test "a new read through a helper and a computed name fails" {
  printf 'fn g() { env_or("BRAIN_VIA_HELPER", 1); std::env::var(key); }\n' >>"$TREE/crates/a/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"BRAIN_VIA_HELPER"* ]]
  [[ "$output" == *"<key>"* ]]
}

@test "a stale allowlist entry fails (the ratchet only shrinks)" {
  printf 'crates/a/src/lib.rs\tBRAIN_GONE\n' >>"$BATS_TEST_TMPDIR/allow"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE"*"BRAIN_GONE"* ]]
}

@test "compile-time cargo variables are not env reads" {
  printf 'const D: &str = env!("CARGO_MANIFEST_DIR");\n' >>"$TREE/crates/a/src/lib.rs"
  gate
  [ "$status" -eq 0 ]
}
