#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements quantized inference that runs on every device by kernel selection for its
# clients. If your team needs expertise in keeping model code free of hard-wired kernel choices then you can
# procure our services by sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-packed-dispatch.sh: a model crate (layer 4-6) that registers, looks up or records
# a packed int8/int4 kernel by name must be allowlisted with a reason; the same text in layer 3 (model::ops's
# crate), a test or a comment is not matched; a stale entry fails.

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

gate() { run bash "$REPO/scripts/gates/check-packed-dispatch.sh" --root "$T" --allow "$ALLOW" --map "$BATS_TEST_TMPDIR/map"; }

@test "the packed kernels in layer 3, a comment or a test module pass" {
  printf 'const K: &[(&str, &str)] = &[("matmul_i8_dyn", kernels::MATMUL_I8_DYN)];\n' >"$T/crates/core/src/lib.rs"
  printf '// dispatches matmul_i8_dyn through Ops\nfn g() {}\n#[cfg(test)]\nmod t { const K: usize = kernels::MAX_ABS_ROW.len(); }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -eq 0 ]
}

@test "registering a packed kernel in a model crate fails and points at Ops" {
  printf 'const K: &[(&str, &str)] = &[("matmul", kernels::MATMUL), ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN)];\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW"*"crates/m/src/lib.rs"*"model::ops::Ops"* ]]
}

@test "a lookup by name and a record through the index-taking helpers fail too" {
  printf 'fn f(g: &Gpu) { let k = g.kernel_index("quant_pack"); }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  printf 'fn f() { s.push(model::dispatch::mm8_rows_off(g, t, scr, w, sw, o, 0, 0, m, k, n)); }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  printf 'fn f() { s.extend(model::int8::quant_rows_steps(g, q, 0, m, k)); }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
}

@test "the dense kernels and Ops calls pass" {
  printf 'const K: &[(&str, &str)] = &[("matmul", kernels::MATMUL), ("matmul_reg3", kernels::MATMUL_REG3)];\nfn f() { let a = ops.act_shared_scratch(s, x, 0, m, k); ops.matmul(s, w, &a, y, 0); }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -eq 0 ]
}

@test "an allowlisted match passes and a stale entry fails" {
  printf 'const K: &[(&str, &str)] = &[("max_abs_row", kernels::MAX_ABS_ROW)];\n' >"$T/crates/m/src/lib.rs"
  printf 'crates/m/src/lib.rs\t1\tlegacy MoE slot\n' >"$ALLOW"
  gate
  [ "$status" -eq 0 ]
  printf 'fn f() {}\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE"* ]]
}
