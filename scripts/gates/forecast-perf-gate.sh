#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# forecast-perf-gate.sh — forecast latency regression gate (pattern:
# wm-perf-gate.sh). Runs each foundation forecaster through `brain perf run`
# (best-of-3, warmup 1, so the first-request model-build is excluded) and checks
# the report against the local baseline with `brain perf gate --floor`.
#
# Dev-box gate, NOT CI: a throttling laptop-class CPU can swing ~2-3x under
# load, so the floor is deliberately generous (0.33 ≈ the wm-perf-gate's 3x
# headroom) - it catches order-of-magnitude pathologies (a serial reduction, a
# lost AVX path, a software-adapter fallback) while absorbing thermal/load
# swings, NOT a 20% thermal delta. On a rested machine or CI, tighten with
# FORECAST_PERF_FLOOR=0.8.
#
# The baseline directory (scripts/gates/forecast-perf-baselines/) is
# gitignored, not committed: its absolute numbers are one machine's snapshot,
# not portable source (scripts/gates/check-large-files.sh rule 2). Capture it
# locally on a rested machine via --update, then hand-review the diff.
#
# Checkpoints via the gate's own variables (a model with unset ones is
# SKIPPED, not failed); `brain perf run` takes them in its --target spec:
#   FORECAST_PERF_KRONOS_TOKENIZER + FORECAST_PERF_KRONOS_DECODER   kronos  (checkpoint dirs)
#   FORECAST_PERF_CHRONOS2                                          chronos2 (.weights)
#   FORECAST_PERF_FINCAST                                           fincast  (.weights)
# The forecast horizon is FORECAST_PERF_HORIZON (default 12), passed as --horizon.
#
# Usage: scripts/gates/forecast-perf-gate.sh [--update]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

source scripts/gates/perf-gate-seed.sh
BIN=./target/release/brain
BASE=scripts/gates/forecast-perf-baselines
FLOOR="${FORECAST_PERF_FLOOR:-0.33}"
INPUT="${FORECAST_PERF_INPUT:-96}"
export BRAIN_DEVICE="${BRAIN_DEVICE:-cpu}"
HORIZON="${FORECAST_PERF_HORIZON:-12}"

[ -x "$BIN" ] || { echo "build first: make build/release"; exit 2; }
update=0; [ "${1:-}" = "--update" ] && update=1
fail=0

gate_one() { # name target
  local name="$1" target="$2" cand
  cand="$(mktemp)"
  seed_args=(); [ "$update" -eq 1 ] || baseline_seed_args seed_args "$BASE/$name.json"
  if ! "$BIN" perf run latency --target "$target" --best-of 3 --warmup 1 --input "$INPUT" --horizon "$HORIZON" "${seed_args[@]}" --out "$cand" >/dev/null 2>&1; then
    echo "  SKIP $name (perf run failed — weights missing or unreadable)"; rm -f "$cand"; return
  fi
  if [ "$update" -eq 1 ]; then
    mkdir -p "$BASE"; cp "$cand" "$BASE/$name.json"; echo "  updated baseline: $name"
  elif [ ! -f "$BASE/$name.json" ]; then
    echo "  NO BASELINE for $name (run: scripts/gates/forecast-perf-gate.sh --update)"; fail=1
  elif "$BIN" perf gate "$cand" --baseline "$BASE/$name.json" --floor "$FLOOR" >/dev/null 2>&1; then
    echo "  PASS $name (>= ${FLOOR} of baseline)"
  else
    echo "  FAIL $name — regressed below ${FLOOR} of baseline:"
    "$BIN" perf gate "$cand" --baseline "$BASE/$name.json" --floor "$FLOOR" 2>&1 | sed 's/^/    /'
    fail=1
  fi
  rm -f "$cand"
}

if [ -n "${FORECAST_PERF_KRONOS_TOKENIZER:-}" ] && [ -n "${FORECAST_PERF_KRONOS_DECODER:-}" ]; then
  gate_one kronos "kronos:${FORECAST_PERF_KRONOS_TOKENIZER}:${FORECAST_PERF_KRONOS_DECODER}"
else
  echo "  SKIP kronos (set FORECAST_PERF_KRONOS_TOKENIZER + FORECAST_PERF_KRONOS_DECODER)"
fi
if [ -n "${FORECAST_PERF_CHRONOS2:-}" ]; then gate_one chronos2 "chronos2:${FORECAST_PERF_CHRONOS2}"; else echo "  SKIP chronos2 (set FORECAST_PERF_CHRONOS2)"; fi
if [ -n "${FORECAST_PERF_FINCAST:-}" ];  then gate_one fincast  "fincast:${FORECAST_PERF_FINCAST}";   else echo "  SKIP fincast (set FORECAST_PERF_FINCAST)"; fi

if [ "$fail" -eq 0 ]; then echo "forecast-perf-gate: OK"; else echo "forecast-perf-gate: FAIL"; fi
exit "$fail"
