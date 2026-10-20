#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for command-line and service
# configuration that stays uniform across many model families for its clients.
# If your team needs expertise in model-agnostic configuration design then you
# can procure our services by sending an email to info@swedishembedded.com.
#
# G6: no architecture-prefixed command-line flag (`make check/scripts`).
#
# brain has many models; an option is either a real choice a person makes
# (a weight or KV-cache format, an explicit fp32 pin, a context cap) and is
# then ONE model-agnostic flag handed to every model through the typed
# serving policy (`model::policy`), or it is derived from model metadata, the
# device and the memory plan and is not a flag at all. A flag spelled
# `--<architecture>-<option>` (the architectures are the layer-4 crates of
# crate-layers.map, except the tool and harness crates `eval`, `npu` and
# friends) is therefore refused in production code. The deprecated aliases that
# keep existing launch lines working are listed in
# scripts/gates/model-flags.allow (path<TAB>count<TAB>reason; shrink-only) and
# are removed with their last caller. Comment lines are not scanned.
#
# Usage: scripts/gates/check-model-flags.sh [--root DIR] [--allow FILE] [--map FILE]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow="$here/model-flags.allow"
map="$here/crate-layers.map"
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  --map) map="$2"; shift 2 ;;
  *) echo "check-model-flags: unknown argument $1" >&2; exit 2 ;;
  esac
done
ids="$(awk -F'\t' '!/^#/ && NF >= 2 && $2 == 4 {sub("^brain-", "", $1); print $1}' "$map" |
  grep -Ev '^(eval|npu|llmserve|imaging|recon|mvs|sfm|viewport|wm-core|wm-display)$')"
# A family name without its version digits (qwen3, qwen35 -> qwen) is the same prefix.
ids="$(printf '%s\n%s\n' "$ids" "$(printf '%s\n' "$ids" | sed 's/[0-9]*$//')" | grep . | sort -u | paste -sd'|')"
exec python3 "$here/ratchet.py" --root "$root" --allow "$allow" --name check-model-flags \
  --pattern="--($ids)-[a-z]" \
  --advice "Make it a model-agnostic option of the serving policy (model::policy) or derive it from the memory plan; do not prefix a flag with an architecture."
