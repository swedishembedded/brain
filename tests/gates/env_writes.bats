#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Spec for scripts/gates/check-env-writes.sh: library code must not mutate the
# process environment (two concurrent trainings read each other's temporary
# value once). A write outside test code that is not on the allowlist fails,
# an allowlist entry without a written reason fails, a stale entry fails (the
# list only shrinks), and code a test owns is not counted.
# Each case runs the gate against a throwaway tree, never the real one.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  TREE="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$TREE/crates/a/src" "$TREE/crates/a/tests"
  printf 'fn f() { std::env::set_var("X", "1"); }\n' >"$TREE/crates/a/src/lib.rs"
  printf 'crates/a/src/lib.rs\t1\ta binary main owns its process\n' >"$BATS_TEST_TMPDIR/allow"
}

gate() {
  run bash "$REPO/scripts/gates/check-env-writes.sh" --root "$TREE" --allow "$BATS_TEST_TMPDIR/allow"
}

@test "an allowlisted write passes" {
  gate
  [ "$status" -eq 0 ]
}

@test "a new write in library code fails and points at passing a value down" {
  printf 'fn g() { std::env::remove_var("Y"); }\n' >>"$TREE/crates/a/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW WRITE"*"crates/a/src/lib.rs"* ]]
  [[ "$output" == *"thread"* ]]
}

@test "a write in a file with no entry fails" {
  printf 'fn g() { std::env::set_var("Y", "2"); }\n' >"$TREE/crates/a/src/other.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"other.rs"* ]]
}

@test "an entry with no reason fails" {
  printf 'crates/a/src/lib.rs\t1\n' >"$BATS_TEST_TMPDIR/allow"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NO REASON"* ]]
}

@test "a stale entry fails (the list only shrinks)" {
  printf 'crates/a/src/gone.rs\t1\twas a main\n' >>"$BATS_TEST_TMPDIR/allow"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE"*"gone.rs"* ]]
}

@test "writes in test code are not counted" {
  printf '#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() { std::env::set_var("Z", "1"); }\n}\n' >>"$TREE/crates/a/src/lib.rs"
  printf 'fn t() { std::env::set_var("Z", "1"); }\n' >"$TREE/crates/a/tests/it.rs"
  gate
  [ "$status" -eq 0 ]
}
