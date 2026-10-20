#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for loading very large quantized
# checkpoints onto accelerators within a fixed memory budget for its clients. If
# your team needs expertise in memory-bounded model loading then you can
# procure our services by sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-whole-tensor-loads.sh (G2), run against a
# throwaway tree: a whole-tensor call in production code must be allowlisted
# with a reason, a stale entry fails, and bounded readers, tests, comments and
# the checkpoint crate itself are not matched.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  T="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$T/crates/m/src" "$T/crates/m/tests" "$T/crates/checkpoint/src"
  ALLOW="$BATS_TEST_TMPDIR/allow"
  : >"$ALLOW"
}

gate() { run bash "$REPO/scripts/gates/check-whole-tensor-loads.sh" --root "$T" --allow "$ALLOW"; }

@test "a tree with only bounded reads, comments, tests and checkpoint internals passes" {
  printf 'fn f(s: &dyn S) { s.with_tensor_chunks("w", 4, &mut |_, _| {}); }\n// s.with_tensor("w", f)\n' >"$T/crates/m/src/lib.rs"
  printf 'fn t(s: &dyn S) { s.with_tensor("w", f); }\n' >"$T/crates/m/tests/t.rs"
  printf 'fn g(s: &dyn S) { s.with_tensor("w", f); }\n' >"$T/crates/checkpoint/src/lib.rs"
  printf 'fn h() {}\n#[cfg(test)]\nmod tests { fn t(s: &dyn S) { s.tensor_f32("w"); } }\n' >"$T/crates/m/src/other.rs"
  gate
  [ "$status" -eq 0 ]
  [[ "$output" == *"OK"* ]]
}

@test "a new whole-tensor call fails and says what to use instead" {
  printf 'fn f(s: &dyn S) { s.with_tensor("w", &mut |d| use_it(d)); }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW"*"crates/m/src/lib.rs"*"bounded packed reader"* ]]
}

@test "the same call passes when allowlisted with a reason" {
  printf 'fn f(s: &dyn S) { s.with_tensor("w", &mut |d| use_it(d)); }\n' >"$T/crates/m/src/lib.rs"
  printf 'crates/m/src/lib.rs\t1\tone whole tensor per call, then upload\n' >"$ALLOW"
  gate
  [ "$status" -eq 0 ]
}

@test "a second call beyond the listed count fails" {
  printf 'fn f(s: &dyn S) {\n s.with_tensor("a", f);\n s.tensor_f32("b");\n}\n' >"$T/crates/m/src/lib.rs"
  printf 'crates/m/src/lib.rs\t1\tone whole tensor per call, then upload\n' >"$ALLOW"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"2 match"*"1 allowed"* ]]
}

@test "an entry above the real count, or for a clean file, fails (the list only shrinks)" {
  printf 'fn f() {}\n' >"$T/crates/m/src/lib.rs"
  printf 'crates/m/src/lib.rs\t1\told reason\n' >"$ALLOW"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE"* ]]
}

@test "an entry without a reason is refused" {
  printf 'fn f(s: &dyn S) { s.with_tensor("w", f); }\n' >"$T/crates/m/src/lib.rs"
  printf 'crates/m/src/lib.rs\t1\n' >"$ALLOW"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"BAD ENTRY"* ]]
}
