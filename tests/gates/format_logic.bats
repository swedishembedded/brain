#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for supporting many numeric storage
# formats in one inference and training engine for its clients. If your team
# needs expertise in quantized execution and format-agnostic model code then you
# can procure our services by sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-format-logic.sh (G3): format-specific logic in a
# layer-4 crate must be allowlisted with a reason; the same text in a layer-3
# crate, a test or a comment is not matched; a stale entry fails.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  T="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$T/crates/m/src" "$T/crates/core/src"
  printf '[package]\nname = "brain-m"\n' >"$T/crates/m/Cargo.toml"
  printf '[package]\nname = "brain-core"\n' >"$T/crates/core/Cargo.toml"
  printf 'brain-core\t3\nbrain-m\t4\n' >"$BATS_TEST_TMPDIR/map"
  ALLOW="$BATS_TEST_TMPDIR/allow"
  : >"$ALLOW"
}

gate() { run bash "$REPO/scripts/gates/check-format-logic.sh" --root "$T" --allow "$ALLOW" --map "$BATS_TEST_TMPDIR/map"; }

@test "the same format logic in a layer-3 crate, a comment or a test module passes" {
  printf 'fn f(w: Weight) { match w { Weight::NF4 { .. } => {} _ => {} } }\n' >"$T/crates/core/src/lib.rs"
  printf '// Weight::NF4 is decoded by model::ops\nfn g() {}\n#[cfg(test)]\nmod t { fn x() { dequantize_rows(1); } }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -eq 0 ]
}

@test "matching a Weight variant in a layer-4 crate fails and points at model::ops" {
  printf 'fn f(w: Weight) { match w { Weight::NF4 { .. } => {} _ => {} } }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW"*"crates/m/src/lib.rs"*"model::ops"* ]]
}

@test "a kernel lookup and a codec call fail too" {
  printf 'fn f() { let k = kernels_cuda::get("x"); let q = int4::quantize_weight_q4(a, 1, 2); }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"1 match"* ]]
}

@test "an allowlisted match passes and a stale entry fails" {
  printf 'fn f() { let k = kernels_cuda::get("x"); }\n' >"$T/crates/m/src/lib.rs"
  printf 'crates/m/src/lib.rs\t1\tlegacy lookup\n' >"$ALLOW"
  gate
  [ "$status" -eq 0 ]
  printf 'fn f() {}\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE"* ]]
}
