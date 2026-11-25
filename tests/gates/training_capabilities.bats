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
# no row and a drifted page all fail; every served architecture is a
# serving-certification row with a support level derived from its cells, and
# the support table compiled into brain_arch carries the same levels. Runs
# against a synthetic tree.

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

@test "a certified contract renders in the training-correctness table and its tests are checked" {
  printf 'cert:Exact resume\tcertify\tresumes bit-exactly\tqwen3/tests/t.rs::lora_works\n' >>"$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -eq 0 ]
  grep -qF '| Exact resume | resumes bit-exactly | `qwen3/tests/t.rs::lora_works` |' "$T/docs/reference/training-capabilities.md"
  printf 'cert:Exact resume\tcertify\tresumes bit-exactly\tqwen3/tests/t.rs::gone\n' >>"$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"MISSING TEST"*"cert:Exact resume"*"gone"* ]]
}

# A synthetic registry and catalog: qwen3 and lpips are served, toyx is a toy
# and imgpipe a pipeline; qwen3 certifies every cell the support level needs.
serving_tree() {
  mkdir -p "$T/crates/arch/src" "$T/crates/catalog/src"
  printf 'pub const ARCHS: &[Arch] = &[\n    arch!("qwen3", "Qwen3", Text, LlamaCpp, "brain-qwen3"),\n    arch!("lpips", "LPIPS", Vision, Brain, "brain-lpips"),\n    arch!("toyx", "Toy", Domain::Toy, Source::Toy, "brain-toyx"),\n];\n' >"$T/crates/arch/src/lib.rs"
  printf 'pub fn models() -> Vec<ModelEntry> {\n    vec![\n        ModelEntry { manifest: lpips::caps::manifest },\n        ModelEntry { manifest: resident_qwen3::direct_manifest },\n        ModelEntry { manifest: toyx::caps::manifest },\n        ModelEntry { manifest: imgpipe::caps::manifest },\n    ]\n}\n' >"$T/crates/catalog/src/lib.rs"
  for c in parity serve cancel batch; do printf 'qwen3\t%s\t\tqwen3/tests/t.rs::lora_works\n' "$c" >>"$BATS_TEST_TMPDIR/decl.tsv"; done
  printf 'lpips\tparity\t\tqwen3/tests/t.rs::lora_works\n' >>"$BATS_TEST_TMPDIR/decl.tsv"
}

regenerate() { python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null; }

@test "every served architecture is a serving-certification row with its derived support level" {
  serving_tree
  regenerate
  gate
  [ "$status" -eq 0 ]
  page="$T/docs/reference/training-capabilities.md"
  grep -qE '^\| `qwen3` \|.*\| production \|$' "$page"
  grep -qE '^\| `lpips` \|.*\| preview \|$' "$page"
  ! grep -qE '^\| `toyx` \|' "$page"
  table="$T/crates/arch/src/support_table.rs"
  grep -qF '("qwen3", Support::Production),' "$table"
  grep -qF '("lpips", Support::Preview),' "$table"
  ! grep -qF '"toyx"' "$table"
}

@test "a support table that differs from the matrix fails and --write repairs it" {
  serving_tree
  regenerate
  sed -i 's/("lpips", Support::Preview)/("lpips", Support::Production)/' "$T/crates/arch/src/support_table.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"DRIFT"*"support_table.rs"* ]]
  regenerate
  gate
  [ "$status" -eq 0 ]
}

@test "a serving cell for an architecture the catalog does not serve fails" {
  serving_tree
  printf 'toyx\tserve\t\tqwen3/tests/t.rs::lora_works\n' >>"$BATS_TEST_TMPDIR/decl.tsv"
  regenerate
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NOT SERVED"*"toyx"* ]]
}

@test "a catalog entry that names no architecture fails" {
  serving_tree
  sed -i 's/imgpipe::caps::manifest/mystery::caps::manifest/' "$T/crates/catalog/src/lib.rs"
  regenerate
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"UNRESOLVED CATALOG ENTRY"*"mystery"* ]]
}

@test "a real-weights cell needs every test flagged" {
  serving_tree
  printf 'qwen3\treal\t\tqwen3/tests/t.rs::lora_works\n' >>"$BATS_TEST_TMPDIR/decl.tsv"
  regenerate
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"UNFLAGGED REAL"* ]]
  sed -i 's/^\(qwen3\treal\t.*\)$/\1 (real)/' "$BATS_TEST_TMPDIR/decl.tsv"
  regenerate
  gate
  [ "$status" -eq 0 ]
}

@test "the streamed policy-gradient cells follow the conformance suite's ASYNC_MATRIX both ways" {
  printf '#[tokio::test(flavor = "multi_thread")]\nasync fn streams() {}\n' >"$T/crates/qwen3/tests/s.rs"
  printf 'fn s() { Subject { name: "qwen3", tiers: vec![Tier { name: "bf16", build: 1 }] } }\nconst ASYNC_MATRIX: &[(&str, bool, bool, bool)] = &[\n    ("qwen3", true, false, true),\n];\n' >"$T/crates/rl/tests/conformance.rs"
  printf 'qwen3\tlora_f32\t\tqwen3/tests/t.rs::lora_works\nqwen3\tlora_bf16\t\tqwen3/tests/t.rs::lora_works\nqwen3\tonline_pg\t\tqwen3/tests/s.rs::streams\n' >"$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"UNDECLARED CAPABILITY"*"policy_publication"* ]]
  printf 'qwen3\tscore_centring\t\tqwen3/tests/s.rs::streams\nqwen3\tpolicy_publication\t\tqwen3/tests/s.rs::streams\n' >>"$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"FALSE CAPABILITY"*"score_centring"* ]]
  grep -v score_centring "$BATS_TEST_TMPDIR/decl.tsv" >"$BATS_TEST_TMPDIR/d2" && mv "$BATS_TEST_TMPDIR/d2" "$BATS_TEST_TMPDIR/decl.tsv"
  python3 "$REPO/scripts/gates/training-capabilities.py" --root "$T" --decl "$BATS_TEST_TMPDIR/decl.tsv" --write >/dev/null
  gate
  [ "$status" -eq 0 ]
}
