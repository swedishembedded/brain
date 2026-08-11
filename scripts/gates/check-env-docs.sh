#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Env-var documentation gate (`make check/scripts`).
#
# Which models `brain serve` serves, and how the engine behaves, is configured
# ONLY through `BRAIN_*` environment variables (there is no config file), so an
# undocumented variable is an unreachable feature: the facenet/upscale perf
# targets shipped dead for exactly this reason (`BRAIN_FACENET_WEIGHTS` vs the
# resident's `BRAIN_FACENET_DIR` — nothing forced the names to agree, and no
# reference existed to check against). This gate greps every `BRAIN_*` variable
# read anywhere under crates/ (env::var, env::var_os, and any helper matching
# `fn(...)("BRAIN_X", ...)` such as env_path/env_or) and requires each one to
# be documented in one of three places:
#   1. the user-facing config reference (default home)
#   2. a per-model page                — for a model-specific
#                                       knob that's more natural to explain in
#                                       context (the page must literally spell
#                                       out the variable name, not a shorthand
#                                       like "`_VAE`")
#   3. the internal testing rules      — test-fixture-only and internal
#                                       engine-tuning vars, neither of which
#                                       belong in user-facing docs
#
# Usage: scripts/gates/check-env-docs.sh   (exits non-zero listing every
# undocumented variable, not just the first)
set -u
cd "$(dirname "$0")/../.."

CONFIG_DOC=docs/using/configuration.md
TESTING_DOC=.agents/rules/testing.md
MODELS_DIR=docs/models
fail=0

vars=$(grep -rhoE '\("BRAIN_[A-Z0-9_]+"' crates/*/src 2>/dev/null \
      | grep -oE 'BRAIN_[A-Z0-9_]+' | sort -u)

for v in $vars; do
  if grep -q "$v" "$CONFIG_DOC" 2>/dev/null; then continue; fi
  if grep -q "$v" "$TESTING_DOC" 2>/dev/null; then continue; fi
  if grep -rq "$v" "$MODELS_DIR" 2>/dev/null; then continue; fi
  echo "UNDOCUMENTED: $v (read in crates/ but absent from $CONFIG_DOC, $TESTING_DOC, and $MODELS_DIR/)"
  fail=1
done

if [ "$fail" -ne 0 ]; then
  echo
  echo "Add each variable to $CONFIG_DOC (user-facing), a docs/models/<model>.md"
  echo "page (model-specific), or $TESTING_DOC (test-only / internal tuning)."
  exit 1
fi
echo "check-env-docs: every BRAIN_* variable read in crates/ is documented"
