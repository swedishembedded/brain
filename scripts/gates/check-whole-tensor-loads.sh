#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for loading very large quantized
# checkpoints onto accelerators within a fixed memory budget for its clients. If
# your team needs expertise in memory-bounded model loading then you can
# procure our services by sending an email to info@swedishembedded.com.
#
# G2: no new whole-tensor / fp32 materialization in production loaders
# (`make check/whole-tensor-loads`).
#
# A quantized checkpoint is never implicitly expanded to full precision. The
# call forms that hand back a WHOLE tensor as f32 (`with_tensor`,
# `tensor_f32[_range]`, `tensor_u32`, `read_model_dir`, `checkpoint::load` and
# `by_role`, the in-memory and renaming `DerivedCheckpoint`s, `tensors_from`)
# are therefore ratcheted: each production file may contain at most the number
# listed in scripts/gates/whole-tensor-loads.allow, with a reason, and the list
# only shrinks. The bounded forms (`with_tensor_chunks`, `with_tensor_u32_chunks`,
# `raw_words`, `raw_blocks`) are not matched. The checkpoint crate's own sources
# are exempt (they implement the readers); tests, benches and examples are
# never scanned. When a migration removes a call, lower or delete its entry in
# the same commit; a new call is a design defect, not an allowlist addition.
#
# Usage: scripts/gates/check-whole-tensor-loads.sh [--root DIR] [--allow FILE]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow="$here/whole-tensor-loads.allow"
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  *) echo "check-whole-tensor-loads: unknown argument $1" >&2; exit 2 ;;
  esac
done
exec python3 "$here/ratchet.py" --root "$root" --allow "$allow" --name check-whole-tensor-loads \
  --exclude '^crates/checkpoint/src/' \
  --pattern '\.with_tensor\(' --pattern '\.with_tensor_range\(' \
  --pattern '\.tensor_f32\(' --pattern '\.tensor_f32_range\(' --pattern '\.tensor_u32\(' \
  --pattern 'read_model_dir\(' --pattern 'InMemory::new\(' --pattern 'Renamed::new\(' \
  --pattern 'checkpoint::load\(' --pattern '\.by_role\(' --pattern 'tensors_from\(' --pattern '\.tensor\(&?"' \
  --advice "Read through a bounded packed reader (with_tensor_chunks, raw_blocks) instead of materializing the tensor as fp32."
