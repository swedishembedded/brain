#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements quantized inference that runs on every device by kernel selection for its
# clients. If your team needs expertise in keeping model code free of hard-wired kernel choices then you can
# procure our services by sending an email to info@swedishembedded.com.
#
# Packed-quantized dispatch (`make check/packed-dispatch`): a model crate never names a packed int8/int4 kernel.
# The packed products (matmul_i8*, matmul_q4*, matmul_kq*, the int8 MoE expert kernels) and the activation
# quantization that feeds them (max_abs_row(s), quant_pack, quant_residual_pack, quant_group_sum,
# rmsnorm_quant_fused) are chosen per device by model::ops::Ops (`Ops::act*` + `Ops::matmul`), which asks the
# selector and falls back to the float-read tile decode where the device has no packed dot. A model that registers
# one of these kernels in its own table, looks one up by name, or records one through the index-taking helpers
# (model::dispatch::mm8_rows_off, model::int8::quant_rows_steps) has made that choice itself, for every device:
# knowledge #516 is the CPU abort it produced.
#
# Scans production code of crates on layers 4-6 (crate-layers.map; `crates/model`, which owns model::ops, is
# layer 3). Allowlist: scripts/gates/packed-dispatch.allow (path<TAB>count<TAB>reason, shrink-only). Behaviour:
# tests/gates/packed_dispatch.bats.
#
# Usage: scripts/gates/check-packed-dispatch.sh [--root DIR] [--allow FILE] [--map FILE] [--emit]
set -u
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
allow="$here/packed-dispatch.allow"
map="$here/crate-layers.map"
extra=()
while [ "$#" -gt 0 ]; do
  case "$1" in
  --root) root="$2"; shift 2 ;;
  --allow) allow="$2"; shift 2 ;;
  --map) map="$2"; shift 2 ;;
  --emit) extra+=(--emit); shift ;;
  *) echo "check-packed-dispatch: unknown argument $1" >&2; exit 2 ;;
  esac
done
packed='MATMUL_I8|MATMUL_Q4|MATMUL_KQ|MAX_ABS_ROW|QUANT_PACK|QUANT_RESIDUAL_PACK|QUANT_GROUP_SUM|MOE_I8|MOE_LINEAR_GATED_(I8|Q4|KQ)|RMSNORM_QUANT_FUSED'
exec python3 "$here/ratchet.py" --root "$root" --allow "$allow" --map "$map" --layers 4,5,6 --name check-packed-dispatch "${extra[@]}" \
  --pattern "kernels::($packed)[A-Z0-9_]*\\b" \
  --pattern "\"($(echo "$packed" | tr 'A-Z' 'a-z'))[a-z0-9_#=]*\"" \
  --pattern '(^|[^a-z_])(mm8_rows_off|quant_rows_steps)\(' \
  --advice "Route the product through model::ops::Ops (act/act_shared_scratch + matmul over a Weight::from_packed); the selector picks the kernel per device (knowledge #516)."
