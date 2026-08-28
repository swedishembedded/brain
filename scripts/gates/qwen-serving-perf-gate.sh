#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# qwen-serving-perf-gate.sh — the concurrent-serving-performance regression
# gate a serving-performance audit and a concurrent-request-batching
# investigation both asked for (pattern: forecast-perf-gate.sh): drives the REAL served path
# (`apiserve::router()`, `http:qwen-synth:` target — random weights, real
# kernels/batching/admission, no checkpoint needed) through `brain perf run
# serve` at a fixed concurrency and checks the report against the local
# baseline with `brain perf gate --floor`. NOT `sweep`: a sweep's `curve`
# artifact carries no flat `ttfa_p99`/`output_per_s` field for `perf gate` to
# read (see `crates/perf/src/gate.rs`'s own "nothing was actually gated"
# refusal) — `serve` at one concurrency is the scenario shaped for gating.
#
# Dev-box gate, NOT CI: a shared dev box's absolute numbers drift across a
# session - the floor is deliberately generous (0.5, i.e. half of baseline)
# to absorb that drift while still catching an order-of-magnitude regression
# (e.g. a rewiring that serializes concurrent requests again).
#
# The baseline directory (scripts/gates/qwen-serving-perf-baselines/) is
# gitignored, not committed: its absolute numbers are one machine's snapshot,
# not portable source (scripts/gates/check-large-files.sh rule 2). Capture it
# locally on a rested machine via --update, then hand-review the diff.
#
# Needs a real tokenizer (chat-template rendering is part of the served
# path) via QWEN_TOKENIZER — SKIPPED, not failed, when unset.
#
# Usage: scripts/gates/qwen-serving-perf-gate.sh [--update]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BIN=./target/release/brain
BASE=scripts/gates/qwen-serving-perf-baselines
FLOOR="${QWEN_SERVING_PERF_FLOOR:-0.5}"
SHAPE="${QWEN_SERVING_PERF_SHAPE:-28x1024x16x151936}" # Qwen3-0.6B's real shape
export BRAIN_DEVICE="${BRAIN_DEVICE:-cpu}"

[ -x "$BIN" ] || { echo "build first: make build/release"; exit 2; }
if [ -z "${QWEN_TOKENIZER:-}" ]; then
  echo "SKIP qwen-serving-perf-gate (set QWEN_TOKENIZER to a real tokenizer.json)"
  exit 0
fi

update=0; [ "${1:-}" = "--update" ] && update=1
name="qwen-synth-${SHAPE}-${BRAIN_DEVICE}"
cand="$(mktemp)"

# NOT --smoke: `brain perf gate` REFUSES a smoke-run candidate outright
# ("not a measurement") — `--requests`/`--warmup` set explicitly instead, to
# stay fast without tripping that refusal. Concurrency 2, not 1: the exact
# shape the concurrent-request-batching investigation cared about (does a
# second concurrent request cost far less than a second solo run).
if ! "$BIN" perf run serve --target "http:qwen-synth:${SHAPE}:${QWEN_TOKENIZER}" \
    --workload chat --input 24 --output 12 --requests 2 --warmup 1 --concurrency 2 --out "$cand" >/dev/null 2>&1; then
  echo "FAIL $name (perf run itself failed)"; rm -f "$cand"; exit 1
fi

fail=0
if [ "$update" -eq 1 ]; then
  mkdir -p "$BASE"; cp "$cand" "$BASE/$name.json"
  echo "updated baseline: $BASE/$name.json"
elif [ ! -f "$BASE/$name.json" ]; then
  echo "NO BASELINE for $name (run: scripts/gates/qwen-serving-perf-gate.sh --update)"; fail=1
elif "$BIN" perf gate "$cand" --baseline "$BASE/$name.json" --floor "$FLOOR" >/dev/null 2>&1; then
  echo "PASS $name (>= ${FLOOR} of baseline)"
else
  echo "FAIL $name — regressed below ${FLOOR} of baseline:"
  "$BIN" perf gate "$cand" --baseline "$BASE/$name.json" --floor "$FLOOR" 2>&1 | sed 's/^/  /'
  fail=1
fi
rm -f "$cand"

[ "$fail" -eq 0 ] && echo "qwen-serving-perf-gate: OK"
exit "$fail"
