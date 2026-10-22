#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements fast GEMV and tensor-core GEMM kernels for every quantized
# weight format for its clients. If your team needs expertise in bandwidth-bound decode and
# batched prefill over stored low-precision weights then you can procure our services by
# sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-tier-gemm.sh: every DecodeFormat has a coverage row; a gap must be
# recorded and a recorded gap that no longer exists fails; named kernels must exist.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  T="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$T/crates/kernels/src" "$T/crates/kernels/wgsl" "$T/crates/kernels-cuda/cu"
  printf 'pub enum DecodeFormat {\n    Bf16 = 0,\n    Nf4 = 1,\n}\n' >"$T/crates/kernels/src/template.rs"
  : >"$T/crates/kernels/wgsl/gemv_a.wgsl"
  : >"$T/crates/kernels-cuda/cu/mma_a.cu"
  MAP="$BATS_TEST_TMPDIR/map"
  ALLOW="$BATS_TEST_TMPDIR/allow"
  printf 'F32\tgemv_a\tlibrary\nBf16\tgemv_a\tlibrary\nNf4\tgemv_a\tnative:mma_a\n' >"$MAP"
  : >"$ALLOW"
}

gate() { run bash "$REPO/scripts/gates/check-tier-gemm.sh" --root "$T" --map "$MAP" --allow "$ALLOW"; }

@test "full coverage with native decode-mma for the quantized format passes and prints the table" {
  gate
  [ "$status" -eq 0 ]
  [[ "$output" == *"Nf4"*"native:mma_a"* ]]
}

@test "a format without a row fails" {
  printf 'F32\tgemv_a\tlibrary\nBf16\tgemv_a\tlibrary\n' >"$MAP"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"Nf4: weight format has no row"* ]]
}

@test "an unrecorded gap fails and a recorded one passes" {
  printf 'F32\tgemv_a\tlibrary\nBf16\t-\tlibrary\nNf4\tgemv_a\tlibrary\n' >"$MAP"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW GAP Bf16 gemv"* ]]
  [[ "$output" == *"NEW GAP Nf4 fused"* ]]
  printf 'Bf16\tgemv\tno kernel\nNf4\tfused\tscratch\n' >"$ALLOW"
  gate
  [ "$status" -eq 0 ]
}

@test "a recorded gap that no longer exists is stale" {
  printf 'Nf4\tfused\tscratch\n' >"$ALLOW"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE Nf4 fused"* ]]
}

@test "a missing kernel file fails" {
  printf 'F32\tgemv_a\tlibrary\nBf16\tgemv_a\tlibrary\nNf4\tgemv_zz\tnative:mma_a\n' >"$MAP"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"gemv kernel 'gemv_zz' does not exist"* ]]
}
