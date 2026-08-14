#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Heavy, opt-in end-to-end validation against REAL model weights: brain's residency
# scheduler (batching/eviction, live Stats counters) plus the generate -> detect ->
# annotate demo actually producing images over D-Bus.
#
# The generate/detect half IS `examples/dbus/detect_pipeline.py` — this file drives
# it directly rather than maintaining a second, drifting copy. The scheduler math
# (batching/eviction) is `scheduler_asserts.py`, next to this file.
#
# Needs BRAIN_E2E=1, a GPU, and real z-image AND yolo weights (detect_pipeline.py
# hard-requires both — it has no "skip detection" mode, unlike the hand-written
# driver this file used to run instead), so this is NOT part of the fast harness:
# `tests/e2e/examples.bats` runs the SAME examples against the weight-free mock
# model instead, on every `make test/e2e`. Run:
#
#   BRAIN_E2E=1 \
#   BRAIN_S3DIT_DIT=... BRAIN_S3DIT_VAE=... BRAIN_S3DIT_QWEN=... BRAIN_S3DIT_TOKENIZER=... \
#   BRAIN_YOLOV8=/path/to/brain-yolov8.weights \
#   make test/e2e/scheduler
#
# SAFETY: the server is started as a background job of the dbus-run-session
# subshell and killed by that same subshell via its own recorded $SRV pid --
# never pkill.

setup_file() {
  [ "${BRAIN_E2E:-0}" = "1" ] || skip "set BRAIN_E2E=1 to run the heavy end-to-end scheduler test"
  command -v busctl >/dev/null || skip "busctl not available"
  command -v dbus-run-session >/dev/null || skip "dbus-run-session not available"
  python3 -c "import jeepney" 2>/dev/null || skip "python jeepney not installed (pip install -e brain-py)"
  [ -n "${BRAIN_S3DIT_DIT:-}" ] || skip "set BRAIN_S3DIT_* to the z-image weights"
  [ -n "${BRAIN_YOLOV8:-}" ] || skip "set BRAIN_YOLOV8 (detect_pipeline.py hard-requires both models)"
  command -v nvidia-smi >/dev/null || skip "no GPU (nvidia-smi)"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BIN="${BRAIN_BIN:-$REPO/target/release/brain}"
  [ -x "$BIN" ] || skip "build first: cargo build --release -p brain-cli ($BIN missing)"
  export BIN
}

@test "scheduler e2e: generate + detect (examples/dbus/detect_pipeline.py) + batching + eviction" {
  # CPU encoder (unset ENCODER_GPU) so each z-image instance is DiT-on-one-GPU +
  # encoder-in-RAM — several sizes then fit across the GPUs and the extra one is
  # evicted, with no single-card overcommit. --reserve-gb keeps headroom for
  # activations.
  run env -u BRAIN_S3DIT_ENCODER_GPU OUT="${OUT:-/tmp/brain_e2e}" SIZE="${SIZE:-256}" STEPS="${STEPS:-8}" BATCH_N="${BATCH_N:-4}" \
    timeout 1800 dbus-run-session -- bash -c "
      '$BIN' serve --dbus --reserve-gb 4 2>/tmp/brain_e2e_srv.log &
      SRV=\$!
      for i in \$(seq 1 60); do busctl --user list 2>/dev/null | grep -q com.swedishembedded.Brain1 && break; sleep 0.3; done
      python3 -u '$REPO/examples/dbus/detect_pipeline.py' \
        && python3 -u '$REPO/tests/e2e/scheduler_asserts.py' --model z-image --size \"\$SIZE\" --steps \"\$STEPS\" --batch-n \"\$BATCH_N\"
      RC=\$?
      kill \$SRV 2>/dev/null
      exit \$RC
    "

  echo "=== driver output ==="
  echo "$output"
  echo "=== server log (tail) ==="
  tail -5 /tmp/brain_e2e_srv.log 2>/dev/null || true

  [ "$status" -eq 0 ]
  [[ "$output" == *"batching=PASS"* ]]
  [[ "$output" == *"eviction=PASS"* ]]
}
