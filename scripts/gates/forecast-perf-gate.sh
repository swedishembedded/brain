#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# forecast-perf-gate.sh — forecast latency regression gate (pattern:
# wm-perf-gate.sh). Runs each foundation forecaster through `brain perf run`
# (best-of-3, warmup 1, so the first-request model-build is excluded) and checks
# the report against the committed baseline with `brain perf gate --floor`.
#
# Dev-box gate, NOT CI: this 155H throttles ~2-3x under load, so the floor is
# deliberately generous (0.33 ≈ the wm-perf-gate's 3x headroom) — it catches
# order-of-magnitude pathologies (a serial reduction, a lost AVX path, a
# software-adapter fallback) while absorbing thermal/load swings, NOT a 20%
# thermal delta. On a rested machine or CI, tighten with FORECAST_PERF_FLOOR=0.8.
# Capture baselines on a rested machine via --update, then hand-review the diff.
#
# Weights via env (a model with unset weights is SKIPPED, not failed):
#   BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER   kronos  (checkpoint dirs)
#   BRAIN_CHRONOS2                                  chronos2 (.weights)
#   BRAIN_FINCAST                                   fincast  (.weights)
#
# Usage: scripts/gates/forecast-perf-gate.sh [--update]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BIN=./target/release/brain
BASE=scripts/gates/forecast-perf-baselines
FLOOR="${FORECAST_PERF_FLOOR:-0.33}"
INPUT="${FORECAST_PERF_INPUT:-96}"
export BRAIN_DEVICE="${BRAIN_DEVICE:-cpu}"
export BRAIN_FORECAST_HORIZON="${BRAIN_FORECAST_HORIZON:-12}"

[ -x "$BIN" ] || { echo "build first: make release"; exit 2; }
update=0; [ "${1:-}" = "--update" ] && update=1
fail=0

gate_one() { # name target
  local name="$1" target="$2" cand
  cand="$(mktemp)"
  if ! "$BIN" perf run latency --target "$target" --best-of 3 --warmup 1 --input "$INPUT" --out "$cand" >/dev/null 2>&1; then
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

if [ -n "${BRAIN_KRONOS_TOKENIZER:-}" ] && [ -n "${BRAIN_KRONOS_DECODER:-}" ]; then
  gate_one kronos "kronos:${BRAIN_KRONOS_TOKENIZER}:${BRAIN_KRONOS_DECODER}"
else
  echo "  SKIP kronos (set BRAIN_KRONOS_TOKENIZER + BRAIN_KRONOS_DECODER)"
fi
if [ -n "${BRAIN_CHRONOS2:-}" ]; then gate_one chronos2 "chronos2:${BRAIN_CHRONOS2}"; else echo "  SKIP chronos2 (set BRAIN_CHRONOS2)"; fi
if [ -n "${BRAIN_FINCAST:-}" ];  then gate_one fincast  "fincast:${BRAIN_FINCAST}";   else echo "  SKIP fincast (set BRAIN_FINCAST)"; fi

if [ "$fail" -eq 0 ]; then echo "forecast-perf-gate: OK"; else echo "forecast-perf-gate: FAIL"; fi
exit "$fail"
