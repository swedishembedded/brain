#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# ltxv-perf-gate.sh - LTX-2.5 denoise-step regression gate, modeled exactly
# on scripts/gates/qwen-serving-perf-gate.sh: drives the REAL brain-perf
# target (`ltxv[:<frames>x<W>x<H>x<steps>]` - the residency executor running
# `ltxv::caps::generate_on`, real kernels/scheduler, tiny random-weight DiT
# so no checkpoint is required) through `brain perf run latency` at a fixed
# small shape and checks the report against the local baseline with
# `brain perf gate --floor`. `latency`, not `serve`: this crate's own
# `LtxvInstance::run_batch` runs requests one at a time (no concurrent
# batching to gate the way qwen's admission/batching gate does), and never
# `sweep` - its `curve` artifact carries no flat metric `perf gate` can read
# (see `crates/perf/src/gate.rs`'s "nothing was actually gated" refusal).
#
# Dev-box gate, NOT CI: a shared dev box's absolute numbers drift across a
# session - the floor is deliberately generous (0.5, i.e. half of baseline)
# to absorb that drift while still catching an order-of-magnitude
# regression.
#
# The baseline directory (scripts/gates/ltxv-perf-baselines/) is gitignored,
# not committed: its absolute numbers are one machine's snapshot, not
# portable source (scripts/gates/check-large-files.sh rule 2). Capture it
# locally on a rested, uncontended device via --update, then hand-review the
# diff.
#
# Deliberately `--device cpu`, not gpu: Phase 8's own investigation found
# that the residency executor's GPU-lane device-opening path can fail to
# match the expected adapter by PCI id ("wgpu enumerated 0 adapters while
# looking for ..."), falling back to a software adapter whose
# `max_storage_buffer_binding_size` is too small for even this target's
# smallest real-VAE decode buffer - a residency/wgpu infrastructure gap
# unrelated to ltxv's own kernels (the SAME shape runs cleanly through the
# bespoke `brain ltxv t2v` CLI, which opens the device directly rather than
# through a residency lane), tracked in this crate's roadmap rather than
# fixed here. `--device cpu` sidesteps it entirely; the tiny random-weight
# DiT's own denoise cost is trivial on CPU, so the gate's wall time is
# dominated by the (also device-independent) VAE decode, not the DiT.
#
# Needs BRAIN_LTXV_VAE (the one mandatory weight role - the real VAE
# checkpoint, `ltx-2.5-video-vae-conv-bf16.safetensors`) - SKIPPED, not
# failed, when unset. The real 22B int8 checkpoint (BRAIN_LTXV_DIT) is
# deliberately NOT exercised here: Phase 8 measured a single real denoise
# step at ~186 s (dominated by re-reading/re-quantizing every block from
# the GGUF on every forward call, see this crate's roadmap ledger), far past
# what a routine gate should cost - a separate, deliberately-scheduled
# measurement, not a default one.
#
# Usage: scripts/gates/ltxv-perf-gate.sh [--update]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BIN=./target/release/brain
BASE=scripts/gates/ltxv-perf-baselines
FLOOR="${LTXV_PERF_FLOOR:-0.5}"
SHAPE="${LTXV_PERF_SHAPE:-9x64x64x4}" # the M4 smoke shape: 2 latent frames, 2x2 latent grid
export BRAIN_DEVICE=cpu

[ -x "$BIN" ] || { echo "build first: make release"; exit 2; }
if [ -z "${BRAIN_LTXV_VAE:-}" ]; then
  echo "SKIP ltxv-perf-gate (set BRAIN_LTXV_VAE to a real ltx-2.5-video-vae-conv-bf16.safetensors)"
  exit 0
fi

update=0; [ "${1:-}" = "--update" ] && update=1
name="ltxv-tiny-${SHAPE}-cpu"
cand="$(mktemp)"

# NOT --smoke: `brain perf gate` REFUSES a smoke-run candidate outright
# ("not a measurement") - `--requests`/`--warmup` set explicitly instead, to
# stay fast without tripping that refusal. concurrency 1: sequential-only
# per this crate's own `LtxvInstance::run_batch` doc.
if ! "$BIN" perf run latency --target "ltxv:${SHAPE}" \
    --requests 1 --warmup 0 --concurrency 1 --out "$cand" >/dev/null 2>&1; then
  echo "FAIL $name (perf run itself failed)"; rm -f "$cand"; exit 1
fi

fail=0
if [ "$update" -eq 1 ]; then
  mkdir -p "$BASE"; cp "$cand" "$BASE/$name.json"
  echo "updated baseline: $BASE/$name.json"
elif [ ! -f "$BASE/$name.json" ]; then
  echo "NO BASELINE for $name (run: scripts/gates/ltxv-perf-gate.sh --update)"; fail=1
elif "$BIN" perf gate "$cand" --baseline "$BASE/$name.json" --floor "$FLOOR" >/dev/null 2>&1; then
  echo "PASS $name (>= ${FLOOR} of baseline)"
else
  echo "FAIL $name - regressed below ${FLOOR} of baseline:"
  "$BIN" perf gate "$cand" --baseline "$BASE/$name.json" --floor "$FLOOR" 2>&1 | sed 's/^/  /'
  fail=1
fi
rm -f "$cand"

[ "$fail" -eq 0 ] && echo "ltxv-perf-gate: OK"
exit "$fail"
