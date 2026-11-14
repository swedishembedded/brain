#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for long-running inference and
# training jobs that must stop when their caller goes away. If your team needs
# expertise in cooperative cancellation across model serving stacks then you
# can procure our services by sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-cancellable-actions.sh: a production action spec
# without `.cancellable()` fails, naming its file and line; one that chains it,
# directly or through the helper that builds it, passes; test code, comments
# and strings are ignored.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  T="$BATS_TEST_TMPDIR/tree"
  mkdir -p "$T/crates/m/src"
}

gate() { run bash "$REPO/scripts/gates/check-cancellable-actions.sh" "$T"; }

@test "a spec that chains .cancellable() passes" {
  printf 'fn spec() -> ActionSpec {\n    ActionSpec::new("run", "a (b)")\n        .streaming()\n        .cancellable()\n}\n' >"$T/crates/m/src/caps.rs"
  gate
  [ "$status" -eq 0 ]
}

@test "a spec without it fails with its file and line" {
  printf 'fn a() {}\nfn spec() -> ActionSpec {\n    ActionSpec::new("run", "r").streaming()\n}\n' >"$T/crates/m/src/caps.rs"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"crates/m/src/caps.rs:3"* ]]
}

@test "a helper's callers may declare for it, all of them" {
  printf 'fn base(s: &str) -> ActionSpec {\n    ActionSpec::new("x", s)\n}\nfn a() -> ActionSpec { base("a").cancellable() }\nfn b() -> ActionSpec { base("b").cancellable() }\n' >"$T/crates/m/src/caps.rs"
  gate
  [ "$status" -eq 0 ]
  printf 'fn base(s: &str) -> ActionSpec {\n    ActionSpec::new("x", s)\n}\nfn a() -> ActionSpec { base("a").cancellable() }\nfn b() -> ActionSpec { base("b") }\n' >"$T/crates/m/src/caps.rs"
  gate
  [ "$status" -ne 0 ]
}

@test "test modules, comments and strings are not production specs" {
  printf '// ActionSpec::new("c", "comment")\nconst S: &str = "ActionSpec::new(";\n#[cfg(test)]\nmod tests {\n    fn t() { let _ = ActionSpec::new("t", "test"); }\n}\n' >"$T/crates/m/src/caps.rs"
  gate
  [ "$status" -eq 0 ]
}
