#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# BRAIN_DEVICE single-source-of-truth gate (`make check/scripts`).
#
# Two independent parsers for --device/BRAIN_DEVICE used to exist: the STRONG
# one (`DeviceSpec::parse` + `resolve`, the full cpu|gpu|npu|vulkan|wgpu grammar
# with indices/ranges) that `crates/cli`'s `select_backend` used for `--device`,
# and a WEAK one (`gpu_core::resolve_backend_name`'s bare "cpu"/"vulkan" string
# match, defaulting everything else - `gpu0`, `npu`, `cpu0-7`, ... - to "just
# use wgpu, ambient card") that every OTHER caller took for a bare
# `BRAIN_DEVICE` (every test binary, every library caller that never went
# through the CLI). `BRAIN_DEVICE=gpu0 cargo test` silently meant "wgpu,
# whatever card is ambient" - not gpu0 specifically. See A4 in
# .agents/roadmap/dtype.md.
#
# The weak ladder is gone: every non-CLI caller now goes through
# `gpu_core::ambient_compute_set()`, which resolves `BRAIN_DEVICE` with the
# SAME strong parser `--device` uses. This gate keeps that fix real by
# asserting `BRAIN_DEVICE` is read via `std::env::var` in exactly ONE file
# under crates/gpu-core/src/ (the canonical resolver) and NOWHERE else under
# any crates/*/src/ - i.e. nothing re-derives its own second opinion.
#
# Known, accepted, DOCUMENTED exception (not fixed by this gate - a later
# phase's job, see the A4 note above): crates/cli/src/qwen_cli.rs's
# want_npu() re-reads BRAIN_DEVICE as a belt-and-suspenders check, because the
# NPU_REQUESTED sidecar it also consults was not fully trusted. Deleting that
# sidecar (phase C3) is what lets this exception be removed.
#
# Usage: scripts/gates/check-device-env-single-source.sh
set -u
cd "$(dirname "$0")/../.."

EXCEPTIONS=(
  "crates/cli/src/qwen_cli.rs"
)

is_exception() {
  local f="$1" ex
  for ex in "${EXCEPTIONS[@]}"; do
    [ "$f" = "$ex" ] && return 0
  done
  return 1
}

# `std::env::var("BRAIN_DEVICE")` - a READ. Deliberately does not match
# `std::env::set_var(...)` (test-only device overrides) or `var_os`, neither
# of which is the ladder this gate is guarding against.
hits=$(grep -rl 'env::var(\s*"BRAIN_DEVICE"' crates/*/src 2>/dev/null || true)

fail=0
canonical=""
for f in $hits; do
  if is_exception "$f"; then
    continue
  fi
  case "$f" in
    crates/gpu-core/src/*)
      if [ -n "$canonical" ] && [ "$canonical" != "$f" ]; then
        echo "MULTIPLE gpu-core files read BRAIN_DEVICE: $canonical and $f"
        fail=1
      fi
      canonical="$f"
      ;;
    *)
      echo "UNEXPECTED BRAIN_DEVICE READ: $f (must go through gpu_core::ambient_compute_set() instead)"
      fail=1
      ;;
  esac
done

if [ -z "$canonical" ]; then
  echo "BRAIN_DEVICE is not read anywhere under crates/gpu-core/src/ -- expected exactly one canonical reader"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "BRAIN_DEVICE must be read in exactly one place: crates/gpu-core/src/devices.rs's"
  echo "ambient_compute_set(). Every other caller should read gpu_core::ambient_compute_set()"
  echo "(or the CLI's own gpu_core::publish_compute_set()-fed compute_set()) instead of"
  echo "re-deriving its own resolution of the env var."
  exit 1
fi
echo "check-device-env-single-source: BRAIN_DEVICE is read in exactly one place (${canonical}), plus the documented qwen_cli.rs exception"
