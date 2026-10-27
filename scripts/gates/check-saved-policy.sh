#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements memory-bounded fine-tuning across model families for its clients. If your team
# needs expertise in holding a training run's activation memory to a budget, you can procure our services by sending
# an email to info@swedishembedded.com.

# Saved-activation coverage (`make check/saved-policy`): every crate that defines a backward pass lets its training
# run choose what the backward keeps of the forward (`model::saved::SavedPolicy`). A crate qualifies when its sources
# contain one of the shared mechanisms:
#   fn set_saved_policy          a `Model` or a trainer that takes the policy
#   BlockStack                   a trainer that holds its own per-block caches (`model::saved::BlockStack`)
#   Activations                  a graph on the conv-block tape (`vae::blocks::Activations`)
#   recompute_only               a backward that always rebuilds every layer (`model::saved::recompute_only`)
#   fn set_recompute             a vision unit that rebuilds its intermediates (`vision::blocks::Conv`)
# A crate with a backward and none of them is listed in scripts/gates/saved-policy.allow, one
# `crate<TAB>reason` per line: a backward kept only for a gradient check of a component frozen in every training
# path, or a computation with no layer stack to recompute. The gate fails on
#   (a) a crate with a backward, no mechanism and no entry;
#   (b) an entry whose crate now has a mechanism or no backward (the list only shrinks);
#   (c) an entry with no reason.
#
# Usage: scripts/gates/check-saved-policy.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

allow="scripts/gates/saved-policy.allow"
backward='fn (backward|block_backward|layer_backward|block_bwd|layer_bwd|backward_steps)\b'
mechanism='fn set_saved_policy|BlockStack|\bActivations\b|recompute_only|fn set_recompute'
fail=0

declare -A listed
while IFS=$'\t' read -r crate reason; do
  [ -z "$crate" ] && continue
  case "$crate" in \#*) continue ;; esac
  if [ -z "${reason// /}" ]; then
    echo "saved-policy: $allow: $crate has no reason"
    fail=1
  fi
  listed[$crate]=1
done < "$allow"

for dir in crates/*/src; do
  crate="${dir#crates/}"
  crate="${crate%/src}"
  if ! grep -rqE "$backward" "$dir" --include='*.rs'; then
    if [ -n "${listed[$crate]:-}" ]; then
      echo "saved-policy: $crate is listed but defines no backward; delete its entry"
      fail=1
    fi
    continue
  fi
  if grep -rqE "$mechanism" "$dir" --include='*.rs'; then
    if [ -n "${listed[$crate]:-}" ]; then
      echo "saved-policy: $crate is listed but takes the policy; delete its entry"
      fail=1
    fi
    continue
  fi
  if [ -z "${listed[$crate]:-}" ]; then
    echo "saved-policy: $crate defines a backward but takes no saved-activation policy"
    echo "  give its training path one (model::saved), or list it in $allow with the reason it needs none"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "saved-policy: OK (${#listed[@]} listed crates need none)"
