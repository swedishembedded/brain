#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# forecast-parity-gate.sh — the forecasting correctness gates as one suite
# (pattern: parity-gate.sh). Every optimization on the time-series models is
# defended by a fp32-exact parity check:
#
#   1. kronos KV-cache path == the reference growing-window forecast (cosine 1.0)
#   2. kronos shared-prefill sampling == N independent rollouts (bit-identical)
#   3. kronos cross-section batch == serial forecast_cached (bit-identical)
#   4. batched training == per-window finite-difference gradcheck  AND
#      a b-batched step == the mean of the b single-window steps (grads allclose)
#
# 1–3 need the real kronos checkpoints (env below); 4 is self-contained (random
# weights). A model whose checkpoints are unset SKIPS, it does not fail.
#
#   KRONOS_TOKENIZER_DIR + KRONOS_DECODER_DIR   kvcache parity (HF checkpoint dirs)
#   BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER  the bench_cpu parity+speed gates
#
# Usage: scripts/gates/forecast-parity-gate.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export BRAIN_DEVICE="${BRAIN_DEVICE:-cpu}"

fail=0
run() { local desc="$1"; shift; echo "=== $desc ==="; if "$@"; then echo "  PASS: $desc"; else echo "  FAIL: $desc"; fail=1; fi; echo; }

# (4) batched-training gates — self-contained (KronosConfig::tiny random weights).
run "batched training == gradcheck + == mean-of-singles" \
    cargo test --release -q -p brain-kronos --test train_gradcheck batched

# (1) KV-cache == reference forecast — needs KRONOS_TOKENIZER_DIR/KRONOS_DECODER_DIR.
if [ -n "${KRONOS_TOKENIZER_DIR:-}" ] && [ -n "${KRONOS_DECODER_DIR:-}" ]; then
    run "kronos KV-cache == reference forecast (cosine 1.0)" \
        cargo test --release -q -p brain-kronos --test kvcache_parity
else
    echo "=== kronos KV-cache parity: SKIP (set KRONOS_TOKENIZER_DIR + KRONOS_DECODER_DIR) ==="; echo
fi

# (2,3) shared-prefill + cross-section bit-identity — need BRAIN_KRONOS_TOKENIZER/_DECODER.
if [ -n "${BRAIN_KRONOS_TOKENIZER:-}" ] && [ -n "${BRAIN_KRONOS_DECODER:-}" ]; then
    run "kronos shared-prefill + cross-section == serial (bit-identical)" \
        cargo test --release -q -p brain-kronos --test bench_cpu \
        shared_prefill_parity_and_speed crosssection_batch_parity_and_speed -- --ignored
else
    echo "=== kronos shared-prefill/cross-section parity: SKIP (set BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER) ==="; echo
fi

if [ "$fail" -eq 0 ]; then echo "forecast-parity-gate: PASS"; else echo "forecast-parity-gate: FAIL"; fi
exit "$fail"
