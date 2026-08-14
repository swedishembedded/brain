#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# wm-perf-gate.sh — world-model fps regression gate (pattern: vemu's
# perf-gate.sh: best-of-N against committed baselines; hard floors only, since
# this laptop-class 155H throttles and soft deltas would flap).
#
# Dev-box gate, NOT CI: needs out/diamond-breakout.weights (brain diamond import).
# Usage: scripts/gates/wm-perf-gate.sh [--update]   (--update rewrites baselines)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BIN=./target/release/brain
BASE=scripts/gates/wm-perf-baselines.json
WEIGHTS=out/diamond-breakout.weights
RUNS=3
export DISPLAY=

[ -x "$BIN" ] || { echo "build first: make release"; exit 2; }
[ -f "$WEIGHTS" ] || { echo "missing $WEIGHTS (brain diamond import ...)"; exit 2; }

best_ms() { # "device [extra args]" -> best mean ms/frame over RUNS
  local dev="$1" best=999999 ms
  for _ in $(seq "$RUNS"); do
    # shellcheck disable=SC2086
    ms=$($BIN diamond bench --model diamond --weights "$WEIGHTS" --device $dev --frames 15 \
      | sed -n 's/.*ms_per_frame_mean=\([0-9.]*\).*/\1/p')
    ms=${ms%.*}
    [ "$ms" -lt "$best" ] && best=$ms
  done
  echo "$best"
}

NPU_ONNX=out/diamond.onnx
npu_ms=""
if [ -f "$NPU_ONNX" ]; then
  npu_ms=$(best_ms "npu --onnx $NPU_ONNX")
fi
cpu_ms=$(best_ms cpu)
gpu_ms=$(best_ms gpu)
echo "measured best-of-$RUNS: cpu=${cpu_ms}ms gpu=${gpu_ms}ms npu=${npu_ms:-n/a}ms per frame (3 denoise steps)"

if [ "${1:-}" = "--update" ]; then
  printf '{\n  "diamond_cpu_ms_max": %s,\n  "diamond_gpu_ms_max": %s,\n  "diamond_npu_ms_max": %s\n}\n' \
    "$((cpu_ms * 300 / 100))" "$((gpu_ms * 300 / 100))" "$(( ${npu_ms:-0} * 300 / 100 ))" > "$BASE"
  echo "baselines updated (x3 headroom): $(cat "$BASE" | tr -d '\n ')"
  exit 0
fi

[ -f "$BASE" ] || { echo "no baselines; run with --update once"; exit 2; }
cpu_max=$(sed -n 's/.*"diamond_cpu_ms_max": *\([0-9]*\).*/\1/p' "$BASE")
gpu_max=$(sed -n 's/.*"diamond_gpu_ms_max": *\([0-9]*\).*/\1/p' "$BASE")
npu_max=$(sed -n 's/.*"diamond_npu_ms_max": *\([0-9]*\).*/\1/p' "$BASE")

fail=0
[ "$cpu_ms" -le "$cpu_max" ] || { echo "FAIL: cpu ${cpu_ms}ms > baseline ${cpu_max}ms"; fail=1; }
[ "$gpu_ms" -le "$gpu_max" ] || { echo "FAIL: gpu ${gpu_ms}ms > baseline ${gpu_max}ms"; fail=1; }
if [ -n "$npu_ms" ] && [ "${npu_max:-0}" -gt 0 ]; then
  [ "$npu_ms" -le "$npu_max" ] || { echo "FAIL: npu ${npu_ms}ms > baseline ${npu_max}ms"; fail=1; }
fi
[ "$fail" -eq 0 ] && echo "wm-perf-gate: OK"
exit "$fail"
