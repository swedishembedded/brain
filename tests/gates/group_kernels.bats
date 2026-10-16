#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for finding where a training step's
# device time actually goes for its clients. If your team needs expertise in
# GPU kernel profiling and throughput engineering then you can procure our
# services by sending an email to info@swedishembedded.com.

# Spec for scripts/perf/group-kernels.py: the component table accounts for every
# kernel of the stored fixture table, so its total is the table's total.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  FIXTURE="$REPO/scripts/perf/testdata/kernel-table.log"
  GROUP="$REPO/scripts/perf/group-kernels.py"
}

@test "the component totals sum to the kernel table's total" {
  run python3 "$GROUP" "$FIXTURE"
  [ "$status" -eq 0 ]
  [[ "$output" == *"total  "*"10827.0"*"76484"* ]]
}

@test "known kernels land in their component" {
  run python3 "$GROUP" "$FIXTURE"
  [[ "$output" == *"GDN chain"*"4944.0"* ]]
  [[ "$output" == *"library GEMM"*"2648.0"*"21896"* ]]
  [[ "$output" == *"NF4 dequant"*"306.0"* ]]
}

@test "the last table is grouped unless --label picks another" {
  run python3 "$GROUP" --label "step 1" "$FIXTURE"
  [ "$status" -eq 0 ]
  [[ "$output" == *"[step 1]"*"element-wise glue"*"1.0"* ]]
}

@test "a PROBE line inside the table does not cut it short" {
  sed '10i PROBE built_mib=1' "$FIXTURE" > "$BATS_TEST_TMPDIR/t.log"
  run python3 "$GROUP" "$BATS_TEST_TMPDIR/t.log"
  [ "$status" -eq 0 ]
  [[ "$output" == *"total  "*"10827.0"* ]]
}

@test "an unknown kernel is billed to misc, not dropped" {
  sed 's/embedding_gather /brand_new_kernel/' "$FIXTURE" > "$BATS_TEST_TMPDIR/t.log"
  run python3 "$GROUP" "$BATS_TEST_TMPDIR/t.log"
  [ "$status" -eq 0 ]
  [[ "$output" == *"head, optimizer, misc"*"374.0"* ]]
}

@test "input without a table, or with lost rows, fails" {
  run python3 "$GROUP" <<<"PROBE step=1"
  [ "$status" -eq 1 ]
  grep -v library_gemm "$FIXTURE" > "$BATS_TEST_TMPDIR/t.log"
  run python3 "$GROUP" "$BATS_TEST_TMPDIR/t.log"
  [ "$status" -eq 1 ]
  [[ "$output" == *"differs from the table"* ]]
}
