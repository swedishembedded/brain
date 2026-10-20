#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for fitting large models onto
# accelerators by memory budget for its clients. If your team needs expertise
# in memory planning and placement then you can procure our services by
# sending an email to info@swedishembedded.com.
#
# G5: placement accounting charges physical destination bytes
# (`make check/scripts`).
#
# `paramstore::dtype::device_f32_words`/`device_bytes`/`tensor_device_bytes`
# charge every stored element as one f32 (a packed-int8 word as four), and
# helpers such as `fp32_weight_bytes` size a model as if it were fp32. That is
# right only for a path that really expands to fp32 on the device; for a
# tensor that stays packed it over-charges by the packing ratio and makes the
# planner refuse or shard a model that fits. Callers of those fp32-element
# sizing helpers are ratcheted outside paramstore/src/dtype.rs: each file's
# count is listed in scripts/gates/placement-accounting.allow with a reason and
# only shrinks. New placement code sizes tensors from their descriptor's
# physical bytes, never from an element count times four.
#
# Usage: scripts/gates/check-placement-accounting.sh [--root DIR] [--allow FILE]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow="$here/placement-accounting.allow"
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  *) echo "check-placement-accounting: unknown argument $1" >&2; exit 2 ;;
  esac
done
exec python3 "$here/ratchet.py" --root "$root" --allow "$allow" --name check-placement-accounting \
  --exclude '^crates/paramstore/src/dtype\.rs$' \
  --pattern '(device_f32_words|tensor_device_bytes)\(' --pattern 'dtype::device_bytes\(' \
  --pattern '(fp32|f32)_(weight|resident)_bytes\(' --pattern 'weights_fp32_bytes\(' \
  --advice "Charge the tensor's physical destination bytes (its storage descriptor), not an fp32 element count."
