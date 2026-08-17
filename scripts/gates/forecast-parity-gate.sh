#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# forecast-parity-gate.sh - the forecasting correctness gates as one suite
# (pattern: parity-gate.sh). Every optimization on the time-series models is
# defended by a fp32-exact parity check, and the user-facing path is defended
# by an end-to-end one:
#
#   1. batched training == per-window finite-difference gradcheck  AND
#      a b-batched step == the mean of the b single-window steps (grads allclose)
#   2. kronos KV-cache path == the reference growing-window forecast (cosine 1.0)
#   3. kronos shared-prefill sampling == N independent rollouts (bit-identical)
#   4. kronos cross-section batch == serial forecast_cached (bit-identical)
#   5. the CSV boundary: a malformed OHLCV file is rejected with a line number
#   6. end-to-end: the committed example CSV -> a validated Panel -> real
#      weights -> a forecast that beats persistence over 4 rolling origins,
#      and the chart renders
#
# 1 and the weight-free half of 5 are self-contained. 2, 3, 4, 6 need the real
# Kronos checkpoints:
#
#   BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER   the two HF checkpoint dirs
#
# ...which `brain forecast predict` auto-fetches, so "I don't have them" is a
# one-command problem now.
#
# EVERY test is invoked unconditionally, and each decides for itself whether an
# absent checkpoint is a skip (`brain_testutil::skip`) or a failure. That is the
# whole design: this script used to branch on `[ -n "$VAR" ]` and print SKIP,
# which made `BRAIN_REQUIRE_FIXTURES=1` a no-op here - the flag can only turn a
# skip into a failure if the test actually RUNS. A gate that has never once run
# is indistinguishable from a gate that passes, and this one had never run.
#
#   scripts/gates/forecast-parity-gate.sh                       # skips are OK
#   BRAIN_REQUIRE_FIXTURES=1 scripts/gates/forecast-parity-gate.sh   # skips are failures
#
# Usage: scripts/gates/forecast-parity-gate.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export BRAIN_DEVICE="${BRAIN_DEVICE:-cpu}"

fail=0
run() { local desc="$1"; shift; echo "=== $desc ==="; if "$@"; then echo "  PASS: $desc"; else echo "  FAIL: $desc"; fail=1; fi; echo; }

# (1) batched-training gates - self-contained (KronosConfig::tiny random weights).
run "batched training == gradcheck + == mean-of-singles" \
    cargo test --release -q -p brain-kronos --test train_gradcheck batched

# (2) KV-cache == reference forecast.
run "kronos KV-cache == reference forecast (cosine 1.0)" \
    cargo test --release -q -p brain-kronos --test kvcache_parity

# (3,4) shared-prefill + cross-section bit-identity. ONE filter per invocation:
# `cargo test` takes a single positional TESTNAME, so passing both names exits
# with a usage error rather than running anything.
run "kronos shared-prefill == serial (bit-identical)" \
    cargo test --release -q -p brain-kronos --test bench_cpu \
    shared_prefill_parity_and_speed -- --ignored
run "kronos cross-section == serial (bit-identical)" \
    cargo test --release -q -p brain-kronos --test bench_cpu \
    crosssection_batch_parity_and_speed -- --ignored

# (5) the CSV boundary on its own, weight-free: every rejection class, and the
# Panel the parser builds. Runs on any machine, with or without checkpoints.
run "OHLCV CSV validated structurally + semantically at entry" \
    cargo test --release -q -p brain-forecast csv::

# (6) the user-facing path end to end. Skips without checkpoints; FAILS without
# them under BRAIN_REQUIRE_FIXTURES=1.
run "CSV -> panel -> kronos -> beats persistence over rolling origins" \
    cargo test --release -q -p brain-kronos --test csv_forecast_e2e

if [ "$fail" -eq 0 ]; then echo "forecast-parity-gate: PASS"; else echo "forecast-parity-gate: FAIL"; fi
exit "$fail"
