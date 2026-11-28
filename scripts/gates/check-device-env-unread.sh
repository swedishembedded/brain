#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
# BRAIN_DEVICE / BRAIN_BACKEND are read nowhere in library or binary code (`make check/scripts`).
#
# They once chose the hardware of every process no CLI configured (a test binary, a library caller), so a shell
# that exported them, or a node whose cards another job held, sent work to a GPU nobody asked for. A device is now
# asked for explicitly: the CLI's `--device`/`--backend`, a tool's own flags (`gpu_core::select_compute`), or a
# caller's own placement; a process nothing configured runs on the CPU (`gpu_core::ambient_compute_set`). This gate
# keeps the variables from coming back under any crates/*/src/.
#
# Usage: scripts/gates/check-device-env-unread.sh
set -u
cd "$(dirname "$0")/../.."

hits=$(grep -rnE 'env::var(_os)?\(\s*"BRAIN_(DEVICE|BACKEND)"' crates/*/src 2>/dev/null || true)
if [ -n "$hits" ]; then
  echo "$hits"
  echo
  echo "BRAIN_DEVICE and BRAIN_BACKEND choose no device: take it from a --device/--backend flag, select_compute or"
  echo "the caller's placement instead."
  exit 1
fi
echo "check-device-env-unread: BRAIN_DEVICE and BRAIN_BACKEND are read nowhere under crates/*/src"
