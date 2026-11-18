#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for test suites whose green result
# can be trusted for its clients. If your team needs expertise in test-gate
# design and continuous verification then you can procure our services by
# sending an email to info@swedishembedded.com.
#
# Spec for scripts/gates/check-test-skips.sh (`make test/gates/test-skips`).
# Each case runs the gate against a throwaway tree, never the real one:
#   - a file with a bare skip that is not allowlisted fails;
#   - the same file with a matching allowlist entry passes;
#   - an allowlist entry larger than the real count fails (the list only
#     shrinks);
#   - a test that skips through brain_testutil, or carries #[ignore], passes;
#   - a test that opens a real GPU with no device guard before it fails, and
#     one guarded directly or through a helper, an #[ignore]d one and a
#     card_tests! binary pass.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
gate="$here/check-test-skips.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
failures=0

tree="$work/tree"
allow="$work/allow"
src="$tree/crates/a/tests/t.rs"

fresh() {
  rm -rf "$tree"
  mkdir -p "$tree/crates/a/tests"
  : >"$allow"
}

expect() { # expect <pass|fail> <name> [output-substring]
  local want="$1" name="$2" needle="${3:-}" out status
  out="$(bash "$gate" --root "$tree" --allow "$allow" 2>&1)"
  status=$?
  if [ "$want" = pass ] && [ "$status" -ne 0 ]; then
    echo "FAIL: $name: expected pass, got status $status"
    echo "$out"
    failures=$((failures + 1))
  elif [ "$want" = fail ] && [ "$status" -eq 0 ]; then
    echo "FAIL: $name: expected failure, gate passed"
    failures=$((failures + 1))
  elif [ -n "$needle" ] && [[ "$out" != *"$needle"* ]]; then
    echo "FAIL: $name: output lacks '$needle'"
    echo "$out"
    failures=$((failures + 1))
  else
    echo "ok: $name"
  fi
}

fresh
cat >"$src" <<'EOF'
#[test]
fn t() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
}
EOF
expect fail "an env-gated early return that is not allowlisted fails" "crates/a/tests/t.rs"

printf 'crates/a/tests/t.rs\t1\n' >"$allow"
expect pass "the same file with a matching allowlist entry passes"

printf 'crates/a/tests/t.rs\t2\n' >"$allow"
expect fail "an allowlist entry above the real count fails (the list only shrinks)" "STALE"

printf 'crates/a/tests/t.rs\t1\ncrates/gone/tests/x.rs\t1\n' >"$allow"
expect fail "an allowlist entry for a file with no skip fails" "STALE"

fresh
cat >"$src" <<'EOF'
#[test]
fn t() {
    let Some(d) = dir() else { eprintln!("DIR unset; skipping"); return; };
}
#[test]
fn u() {
    let Some(d) = dir() else {
        eprintln!("no CUDA device on this box - skipping");
        return;
    };
}
EOF
expect fail "a bare eprintln skipping + return fails" "NEW SKIP"
printf 'crates/a/tests/t.rs\t2\n' >"$allow"
expect pass "both bare skips allowlisted pass"

fresh
cat >"$src" <<'EOF'
#[test]
#[ignore = "needs a GPU"]
fn t() {}
#[test]
fn u() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    // eprintln!("skipping") in a comment is not a skip
}
#[test]
fn v() {
    eprintln!("eval: sample 3 failed; skipping");
    counter += 1;
}
EOF
expect pass "skip_unavailable-style helpers, #[ignore], comments and non-returning notes pass"

fresh
printf 'fn f() { std::env::var("MOE_SKIP_GPU_TESTS"); }\n' >"$tree/crates/a/tests/t.rs"
mkdir -p "$tree/crates/testutil/src"
printf 'pub fn g() { std::env::var("MOE_SKIP_GPU_TESTS"); }\n' >"$tree/crates/testutil/src/lib.rs"
printf 'crates/a/tests/t.rs\t1\n' >"$allow"
expect pass "the testutil crate, which owns the read, is exempt"

fresh
cat >"$src" <<'EOF'
#[test]
fn t() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let g = Gpu::new_gpu(KERNELS);
}
EOF
expect fail "a test that opens a GPU without a device guard fails" "NEW SKIP"

fresh
cat >"$src" <<'EOF'
fn skip_gpu() -> bool {
    brain_testutil::gpu_unavailable(gpu_core::testgpu::missing_gpu())
}
#[test]
fn t() {
    if brain_testutil::gpu_unavailable(gpu_core::testgpu::missing_gpu()) {
        return;
    }
    let g = Gpu::new_gpu(KERNELS);
}
#[test]
fn u() {
    if skip_gpu() {
        return;
    }
    let g = Gpu::new_wgpu(KERNELS);
}
#[test]
#[ignore = "a benchmark"]
fn bench() {
    let g = WgpuBackend::new(KERNELS);
}
EOF
expect pass "a GPU open behind gpu_unavailable, a helper that calls it, or #[ignore] passes"

fresh
cat >"$src" <<'EOF'
fn t() {
    let g = Gpu::new_gpu(KERNELS);
}
gpu_core::card_tests!(t);
EOF
expect pass "a card_tests! binary, whose card is an argument, passes"

if [ "$failures" -ne 0 ]; then
  echo "test-skips-spec: $failures case(s) failed"
  exit 1
fi
echo "test-skips-spec: OK"
