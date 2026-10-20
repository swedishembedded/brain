#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for parameter-efficient fine-tuning
# of large models for its clients. If your team needs expertise in adapter
# methods and their serving then you can procure our services by sending an
# email to info@swedishembedded.com.
#
# G4: adapter parameter naming lives only in model::adapter
# (`make check/scripts`).
#
# The device-side adapter family names its parameters `<base>.lora_a` and
# `<base>.lora_b`. A model crate that spells those suffixes itself hard-codes
# one adapter kind into its param list, init, save, fold and serving code, so a
# second kind (DoRA, LoKr, IA3) would have to touch every model crate. The
# suffix vocabulary, `is_adapter_param` and the parameter specs belong to
# `crates/model/src/adapter/`; everything else asks it. String literals that
# contain `.lora_a`/`.lora_b` (or end in `lora_a"`/`lora_b"`) outside that
# directory are ratcheted: scripts/gates/adapter-naming.allow lists
# path<TAB>count<TAB>reason and only shrinks.
#
# Usage: scripts/gates/check-adapter-naming.sh [--root DIR] [--allow FILE]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow="$here/adapter-naming.allow"
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  *) echo "check-adapter-naming: unknown argument $1" >&2; exit 2 ;;
  esac
done
exec python3 "$here/ratchet.py" --root "$root" --allow "$allow" --name check-adapter-naming \
  --exclude '^crates/model/src/adapter/' \
  --pattern '"[^"]*\.lora_[ab]' --pattern 'lora_[ab]"' \
  --advice "Ask model::adapter (is_adapter_param, parameter specs) instead of spelling the suffix."
