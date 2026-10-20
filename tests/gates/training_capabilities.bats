#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for documentation of model
# capabilities that cannot drift from the code for its clients. If your team
# needs expertise in capability matrices backed by tests then you can procure
# our services by sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-training-capabilities.sh: the capability page is
# generated from the declaration; a missing or ignored test, a registration with
# no row and a drifted page all fail. Runs against a synthetic tree.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  T="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$T/crates/qwen3/tests" "$T/crates/rl/tests" "$T/crates/sdk/src" "$T/docs/reference"
  printf '#[test]\nfn lora_works() {}\n#[test]\n#[ignore = "slow"]\nfn slow_one() {}\n' >"$T/crates/qwen3/tests/t.rs"
  printf 'fn s() { Subject { name: "qwen3", tiers: vec![Tier { name: "bf16", build: 1 }] } }\n' >"$T/crates/rl/tests/conformance.rs"
  printf 'impl StudyArch for Qwen3 {\n const ID: &'"'"'static str = "qwen3";\n const CHAT_LORA_FINE_TUNE: bool = true;\n fn study_config() {}\n}\n' >"$T/crates/sdk/src/study.rs"
  printf 'qwen3\tlora_f32\t\tqwen3/tests/t.rs::lora_works\nqwen3\tlora_bf16\t\tqwen3/tests/t.rs::lora_works\n' >"$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
}

gate() { run bash "$REPO/scripts/gates/check-training-capabilities.sh" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv"; }

@test "a consistent declaration and page pass" {
  gate
  [ "$status" -eq 0 ]
}

@test "a named test that does not exist fails" {
  printf 'qwen3\tlora_f32\t\tqwen3/tests/t.rs::nope\nqwen3\tlora_bf16\t\tqwen3/tests/t.rs::lora_works\n' >"$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"MISSING TEST"*"nope"* ]]
}

@test "an ignored test needs a (real) or (cuda) flag" {
  printf 'qwen3\tlora_f32\t\tqwen3/tests/t.rs::slow_one\nqwen3\tlora_bf16\t\tqwen3/tests/t.rs::lora_works\n' >"$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"IGNORED TEST"* ]]
  printf 'qwen3\tlora_f32\t\tqwen3/tests/t.rs::slow_one (real)\nqwen3\tlora_bf16\t\tqwen3/tests/t.rs::lora_works\n' >"$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -eq 0 ]
}

@test "a conformance tier with no declared row fails" {
  printf 'qwen3\tlora_f32\t\tqwen3/tests/t.rs::lora_works\n' >"$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"UNDECLARED TIER"*"bf16"* ]]
}

@test "an SDK chat fine-tuner declaration with no lora row fails" {
  printf 'qwen3\tkv_tiers\t\tqwen3/tests/t.rs::lora_works\nqwen3\tlora_bf16\t\tqwen3/tests/t.rs::lora_works\n' >"$BATS_TEST_TMPDIR/decl.tsv"
  sed -i 's/lora_bf16/full_ft/' "$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"UNDECLARED CAPABILITY"*"CHAT_LORA_FINE_TUNE"* ]]
}

@test "a hand-edited page fails and --write repairs it" {
  echo "tampered" >>"$T/docs/reference/training-capabilities.md"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"DRIFT"* ]]
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -eq 0 ]
}
