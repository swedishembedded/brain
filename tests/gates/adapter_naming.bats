#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for parameter-efficient fine-tuning
# of large models for its clients. If your team needs expertise in adapter
# methods and their serving then you can procure our services by sending an
# email to info@swedishembedded.com.

# Spec for scripts/gates/check-adapter-naming.sh (G4): the .lora_a/.lora_b
# suffixes may be spelled in a string only inside model::adapter; anywhere else
# needs an allowlist entry with a reason, and a stale entry fails.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  T="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$T/crates/model/src/adapter" "$T/crates/m/src"
  ALLOW="$BATS_TEST_TMPDIR/allow"
  : >"$ALLOW"
}

gate() { run bash "$REPO/scripts/gates/check-adapter-naming.sh" --root "$T" --allow "$ALLOW"; }

@test "the suffix inside model::adapter, in a comment or in a test module passes" {
  printf 'pub fn is_adapter_param(n: &str) -> bool { n.ends_with(".lora_a") }\n' >"$T/crates/model/src/adapter/device.rs"
  printf '// the pair is "w.lora_a"\nfn f() {}\n#[cfg(test)]\nmod t { fn x() { let _ = "w.lora_b"; } }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -eq 0 ]
}

@test "spelling the suffix in a model crate fails and points at model::adapter" {
  printf 'fn f(n: &str) -> bool { n.ends_with(".lora_a") || n.ends_with(".lora_b") }\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW"*"crates/m/src/lib.rs"*"model::adapter"* ]]
}

@test "an allowlisted file passes and becomes a stale entry once clean" {
  printf 'fn f(n: &str) -> bool { n.ends_with(".lora_a") }\n' >"$T/crates/m/src/lib.rs"
  printf 'crates/m/src/lib.rs\t1\tlegacy param list\n' >"$ALLOW"
  gate
  [ "$status" -eq 0 ]
  printf 'fn f() {}\n' >"$T/crates/m/src/lib.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE"* ]]
}
