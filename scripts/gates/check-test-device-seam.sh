#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for test harnesses that run GPU code on exactly the hardware they are
# asked to, for its clients. If your team needs expertise in backend-matrix testing then you can procure our
# services by sending an email to info@swedishembedded.com.
#
# The test harness's device seam stays one function (`make check/scripts`): BRAIN_TEST_DEVICE is read only by
# `brain_testutil::device::device()` (crates/testutil/src/device.rs), and brain-testutil is only ever a
# dev-dependency, so no library or binary code can reach it. A test binary takes the device through that function;
# the library is told it explicitly (gpu_core::select_compute) and reads no device variable.
#
# Usage: scripts/gates/check-test-device-seam.sh
set -u
cd "$(dirname "$0")/../.."

fail=0
readers=$(grep -rlE 'BRAIN_TEST_DEVICE"|env::var(_os)?\(\s*"BRAIN_TEST_DEVICE' --include=*.rs crates samples 2>/dev/null | grep -v '^crates/testutil/src/device.rs$' || true)
for f in $readers; do
  if grep -qE 'env::var(_os)?\(\s*"BRAIN_TEST_DEVICE' "$f"; then
    echo "READS BRAIN_TEST_DEVICE: $f (take the device from brain_testutil::device::device())"
    fail=1
  fi
done
for toml in crates/*/Cargo.toml samples/*/Cargo.toml samples/*/*/Cargo.toml; do
  [ -f "$toml" ] || continue
  if awk '/^\[/{sec=$0} /^brain-testutil([ .=]|$)/{ if (sec !~ /dev-dependencies/) { print FILENAME": "sec; found=1 } } END{exit !found}' "$toml"; then
    echo "  brain-testutil is a normal dependency there: library code must not reach the test device seam"
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "check-test-device-seam: BRAIN_TEST_DEVICE is read only by brain_testutil::device::device(), a dev-only crate"
