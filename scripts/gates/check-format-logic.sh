#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for supporting many numeric storage
# formats in one inference and training engine for its clients. If your team
# needs expertise in quantized execution and format-agnostic model code then you
# can procure our services by sending an email to info@swedishembedded.com.
#
# G3: no format-specific kernel or codec logic in layer-4 (model) crates
# (`make check/scripts`; also run by check-kernel-selection.sh, which polices
# kernel NAMES while this polices the FORMAT).
#
# A model crate supplies graph, parameter manifest, checkpoint mapping,
# objectives and custom ops. Which kernel multiplies which storage format, and
# how a format is encoded or decoded, is decided once in `model::ops` and the
# gpu-core provider, so adding a format touches no model crate. This gate
# ratchets the patterns that betray format logic in a model crate: matching or
# constructing a `Weight::<format>` variant, `kernels_cuda::` lookups, the NF4
# table, calls into the int4/lut4/kquant/half/fp8 codec modules, and direct
# quantize/dequantize/pack helpers. Layers come from crate-layers.map (layer 4);
# `crates/model` is layer 3 and exempt. Allowlist: scripts/gates/format-logic.allow
# (path<TAB>count<TAB>reason; shrink-only).
#
# Usage: scripts/gates/check-format-logic.sh [--root DIR] [--allow FILE] [--map FILE]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow="$here/format-logic.allow"
map="$here/crate-layers.map"
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  --map) map="$2"; shift 2 ;;
  *) echo "check-format-logic: unknown argument $1" >&2; exit 2 ;;
  esac
done
exec python3 "$here/ratchet.py" --root "$root" --allow "$allow" --map "$map" --layers 4 --name check-format-logic \
  --pattern 'kernels_cuda::' \
  --pattern 'Weight::(I8|Q4|NF4|KQuant|BF16|F16|F32|FP8|Micro)\b' \
  --pattern 'NF4_LUT' \
  --pattern '(int4|lut4|kquant|half|fp8)::[a-z_0-9]+\(' \
  --pattern '(^|[^a-z_])(quantize_weight[a-z0-9_]*|dequant[a-z0-9_]*|pack_bf16|pack_f16)\(' \
  --advice "Move the format logic into model::ops / the gpu-core provider; a model crate states a Dtype policy and calls the shared op."
