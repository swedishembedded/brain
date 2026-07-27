#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# End-to-end validation of brain's residency scheduler over D-Bus, with real models.
#
# Generates an image (z-image), runs object detection (yolo, when configured), and
# validates BATCHING (concurrent requests coalesce) and EVICTION (more resident
# instances than fit force an LRU swap) via the live Stats counters — with timing.
#
# Heavy (loads the 6B z-image weights, minutes to run), so it is opt-in:
#
#   BRAIN_E2E=1 \
#   BRAIN_ZIMAGE_DIT=... BRAIN_ZIMAGE_VAE=... BRAIN_ZIMAGE_QWEN=... BRAIN_ZIMAGE_TOKENIZER=... \
#   [BRAIN_YOLO=/path/to/brain-yolov8.weights] \
#   bats tests/e2e/scheduler.bats

setup() {
  [ "${BRAIN_E2E:-0}" = "1" ] || skip "set BRAIN_E2E=1 to run the heavy end-to-end scheduler test"
  command -v busctl >/dev/null || skip "busctl not available"
  command -v dbus-run-session >/dev/null || skip "dbus-run-session not available"
  python3 -c "import jeepney" 2>/dev/null || skip "python jeepney not installed (pip install jeepney)"
  [ -n "${BRAIN_ZIMAGE_DIT:-}" ] || skip "set BRAIN_ZIMAGE_* to the z-image weights"
  command -v nvidia-smi >/dev/null || skip "no GPU (nvidia-smi)"
  BIN="${BRAIN_BIN:-target/release/brain}"
  [ -x "$BIN" ] || skip "build first: cargo build --release -p brain-cli ($BIN missing)"
  DRIVER="$(dirname "$BATS_TEST_FILENAME")/e2e_scheduler.py"
}

@test "scheduler e2e: generate + detect + batching + eviction over D-Bus" {
  # CPU encoder (unset ENCODER_GPU) so each z-image instance is DiT-on-one-GPU +
  # encoder-in-RAM — several sizes then fit across the GPUs and the extra one is
  # evicted, with no single-card overcommit. reserve keeps headroom for activations.
  run env -u BRAIN_ZIMAGE_ENCODER_GPU OUT="${OUT:-/tmp/brain_e2e}" SIZE="${SIZE:-256}" STEPS="${STEPS:-8}" BATCH_N="${BATCH_N:-4}" \
    timeout 1800 dbus-run-session -- bash -c "
      '$BIN' serve --dbus --reserve-gb 4 2>/tmp/brain_e2e_srv.log &
      SRV=\$!
      for i in \$(seq 1 60); do busctl --user list 2>/dev/null | grep -q com.swedishembedded.Brain1 && break; sleep 0.3; done
      python3 -u '$DRIVER'
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
  [[ "$output" == *"[generate]"* ]]
}
