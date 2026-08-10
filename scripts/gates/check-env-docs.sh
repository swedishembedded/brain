#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Env-var documentation gate (`make check/scripts`).
#
# Which models `brain serve` serves is configured ONLY through `BRAIN_*`
# environment variables (there is no config file), so an undocumented variable
# is an unreachable feature: the facenet/upscale perf targets shipped dead for
# exactly this reason (`BRAIN_FACENET_WEIGHTS` vs the resident's
# `BRAIN_FACENET_DIR` — nothing forced the names to agree, and no reference
# existed to check against). This gate greps every `env::var("BRAIN_…")` read
# in the SERVING crates and requires each variable to appear in
# docs/serving.md's Configuration section, so a new one cannot ship silently.
#
# Scope is the serving layer (cli/apiserve/dbus/npu/residency): model-crate
# internals (BRAIN_ZIMAGE_ENCODER_*, BRAIN_FLUX2_TE_DEVICE, …) are documented
# in their model ledgers instead, per the doc's own note.
#
# Usage: scripts/gates/check-env-docs.sh   (exits non-zero listing every
# undocumented variable, not just the first)
set -u
cd "$(dirname "$0")/../.."

DOC=docs/serving.md
fail=0

vars=$(grep -rhoE 'env::var\("BRAIN_[A-Z0-9_]+"' \
        crates/cli/src crates/apiserve/src crates/dbus/src crates/npu/src crates/residency/src 2>/dev/null \
      | grep -oE 'BRAIN_[A-Z0-9_]+' | sort -u)

for v in $vars; do
  if ! grep -q "$v" "$DOC"; then
    echo "UNDOCUMENTED: $v (read in the serving crates but absent from $DOC)"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo
  echo "Add each variable to $DOC § Configuration (gating, tuning, or server knobs)."
  exit 1
fi
echo "check-env-docs: every serving BRAIN_* var is documented in $DOC"
