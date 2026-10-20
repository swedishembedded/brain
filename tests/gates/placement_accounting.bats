#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for fitting large models onto
# accelerators by memory budget for its clients. If your team needs expertise
# in memory planning and placement then you can procure our services by
# sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-placement-accounting.sh (G5): a caller of an
# fp32-element sizing helper needs an allowlist entry with a reason; the
# helper's own file, comments and tests are not matched; a stale entry fails.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  T="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$T/crates/paramstore/src" "$T/crates/m/src"
  ALLOW="$BATS_TEST_TMPDIR/allow"
  : >"$ALLOW"
}

gate() { run bash "$REPO/scripts/gates/check-placement-accounting.sh" --root "$T" --allow "$ALLOW"; }

@test "the helper's own file, comments and test modules pass" {
  printf 'pub fn device_bytes(d: Option<&str>, n: u64) -> u64 { device_f32_words(d, n) * 4 }\n' >"$T/crates/paramstore/src/dtype.rs"
  printf '// device_bytes(x) counts f32\nfn f() {}\n#[cfg(test)]\nmod t { fn x() { let _ = fp32_weight_bytes(c); } }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -eq 0 ]
}

@test "charging a packed tensor through the fp32 helper fails and says to use physical bytes" {
  printf 'fn plan(r: &R) -> u64 { paramstore::dtype::device_bytes(r.dtype("w"), 8) }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW"*"crates/m/src/lib.rs"*"physical destination bytes"* ]]
}

@test "an allowlisted caller passes and a stale entry fails" {
  printf 'fn plan(c: &C) -> u64 { fp32_weight_bytes(c) }\n' >"$T/crates/m/src/lib.rs"
  printf 'crates/m/src/lib.rs\t1\tlegacy fp32 estimate\n' >"$ALLOW"
  gate
  [ "$status" -eq 0 ]
  printf 'fn plan() {}\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE"* ]]
}
