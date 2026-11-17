#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements model serving engines that fail at test time rather than on a customer's request
# for its clients. If your team needs expertise in keeping a multi-model inference engine's kernel registry
# consistent, you can procure our services by sending an email to info@swedishembedded.com.

# Ops kernel-list coverage (`make check/ops-kernel-lists`): every crate that builds a `model::ops::Ops` in its
# sources (`Ops::new`, `Ops::with_selector`, `Ops::with_providers`) tests each pipeline list it builds one from with
# `model::ops::assert_kernel_list_complete` (or `assert_training_kernel_list_complete`), in `src/` or `tests/`.
# `Ops::new` resolves the façade set by name on a handle a serving engine builds lazily, at a model's first
# request, so a list short of `REQUIRED_KERNELS` passes every other test and panics live (knowledge #507).
# A crate that needs no such test is listed in scripts/gates/ops-kernel-lists.allow, one `crate<TAB>reason` per
# line. The gate fails on
#   (a) a crate that builds an `Ops`, has no completeness assertion and no entry;
#   (b) an entry whose crate now has the assertion or builds no `Ops` (the list only shrinks);
#   (c) an entry with no reason.
#
# Usage: scripts/gates/check-ops-kernel-lists.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

allow="scripts/gates/ops-kernel-lists.allow"
builds='Ops::(new|with_selector|with_providers)\('
asserts='assert_(training_)?kernel_list_complete\('
fail=0

declare -A listed
while IFS=$'\t' read -r crate reason; do
  [ -z "$crate" ] && continue
  case "$crate" in \#*) continue ;; esac
  if [ -z "${reason// /}" ]; then
    echo "ops-kernel-lists: $allow: $crate has no reason"
    fail=1
  fi
  listed[$crate]=1
done < "$allow"

checked=0
for dir in crates/*/src; do
  crate="${dir#crates/}"
  crate="${crate%/src}"
  # A construction on a line that is not a comment (doc comments name these constructors too).
  if ! grep -rhE "$builds" "$dir" --include='*.rs' | grep -qvE '^\s*//'; then
    if [ -n "${listed[$crate]:-}" ]; then
      echo "ops-kernel-lists: $crate is listed but builds no Ops; delete its entry"
      fail=1
    fi
    continue
  fi
  checked=$((checked + 1))
  if grep -rqE "$asserts" "crates/$crate/src" "crates/$crate/tests" --include='*.rs' 2>/dev/null; then
    if [ -n "${listed[$crate]:-}" ]; then
      echo "ops-kernel-lists: $crate is listed but tests its kernel lists; delete its entry"
      fail=1
    fi
    continue
  fi
  if [ -z "${listed[$crate]:-}" ]; then
    echo "ops-kernel-lists: $crate builds a model::ops::Ops but never asserts its kernel list is complete"
    echo "  compose the list with model::ops::extend_kernel_list(.., model::ops::kernel_list()) and test it with"
    echo "  model::ops::assert_kernel_list_complete, or list the crate in $allow with the reason it needs none"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "ops-kernel-lists: OK ($checked crates build an Ops; ${#listed[@]} listed need no test)"
